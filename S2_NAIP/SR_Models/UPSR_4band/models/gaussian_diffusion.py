import enum
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from PIL import Image
from torchvision import transforms

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
        print('sqrt_etas: ', sqrt_etas)
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    PREVIOUS_X = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon

class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL

class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance.
    """

    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf
        
        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev
        self.f = np.ones(self.num_timesteps)
        self.f_prev = np.ones(self.num_timesteps)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # print(self.posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        
        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)

        self.weight_loss_mse = weight_loss_mse

    def q_mean_variance(self, x_start, y, y_hat, un, t):
        """
        Get the distribution q(x_t | x_0, y, y_hat).
        y_tilde = y_hat + f_t * (y - y_hat), f_t = t / T
        x_target = x_start

        q(x_t | x_0, y_tilde) = N(x_t; x_target + eta_t * (y_tilde - x_target), un^2 * kappa^2 * eta_t * I)

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param un: uncertainty map.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        y_tilde = y_hat + _extract_into_tensor(self.f, t, x_start.shape) * (y - y_hat)
        x_target = x_start
        # x_target = (1 - un) * x_start + un * y_hat

        mean = x_target + _extract_into_tensor(self.etas, t, x_target.shape) * (y_tilde - x_target)
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * (self.kappa**2) * (un ** 2)
        log_variance = variance.log()

        return mean, variance, log_variance

    def q_sample(self, x_start, y, y_hat, un, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0, y, y_hat).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param un: the uncertainty map of y.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape == un.shape
        y_tilde = y_hat + _extract_into_tensor(self.f, t, x_start.shape) * (y - y_hat)
        x_target = x_start
        # x_target = (1 - un) * x_start + un * y_hat
        return (
            x_target + _extract_into_tensor(self.etas, t, x_target.shape) * (y_tilde - x_target)
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_target.shape) * un * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, y, y_hat, un, t):
        """
        Compute the mean and variance of the diffusion posterior:

        q(x_{t-1} | x_t, x_0, y, y_hat)
        = N(x_{t-1}; eta_{t-1}/eta_t * x_t + alpha_t/eta_t * x_0 + eta_{t-1} * (y_tilde_{t-1} - y_tilde_t), un^2 * kappa^2 * eta_{t-1}/eta_t * alpha_t * I)

        """
        assert x_start.shape == x_t.shape
        x_target = x_start
        # x_target = (1 - un) * x_start + un * y_hat
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_target
            # + _extract_into_tensor(self.etas_prev, t, x_t.shape) / self.num_timesteps * (y_hat - y)
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape) * un ** 2
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        ) + 2 * torch.log(un)

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, y_hat, un, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t, y, y_hat), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param un: uncertainty.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        model_output = self.encode_first_stage(model(self._scale_input(x_t, un, t), t, **model_kwargs))

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape) * (un ** 2)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape) + 2 * torch.log(un)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, un=un, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, un=un, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, y=y, y_hat=y_hat, un=un, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, y, un, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * un * eps
        ) / (1 - _extract_into_tensor(self.etas, t, x_t.shape) * un)

    def _predict_xstart_from_eps_scale(self, x_t, y, un, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps
        ) / (1 - _extract_into_tensor(self.etas, t, x_t.shape) * un)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, un, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)

    def p_sample(self, model, x, y, y_hat, un, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param y_hat: the [N x C x ...] tensor of improved degraded inputs.
        :param un: the [N x C x ...] tensor of uncertainty values.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            y_hat,
            un,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}

    def p_sample_loop(
        self, y, y_hat, un, model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param y_hat: the [N x C x ...] tensor of improved degraded inputs.
        :param un: the [N x C x ...] tensor of uncertainty values.
        :param model: the model module.
        :param first_stage_model: the autoencoder model
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            y_hat,
            un,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample["sample"]
        return nn.PixelShuffle(self.sf)(final)

    def p_sample_loop_progressive(
            self, y, y_hat, un, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            one_step=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        y = self.encode_first_stage(y)
        y_hat = self.encode_first_stage(y_hat)
        un = self.encode_first_stage(un)
        model_kwargs['un'] = un

        # generating noise
        if noise is None:
            noise = torch.randn_like(y)
        else:
            noise = self.encode_first_stage(noise)

        if noise_repeat:
            noise = noise[0,].repeat(y.shape[0], 1, 1, 1)

        x_sample = self.prior_sample(y, y_hat, un, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * y.shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    x_sample,
                    y, y_hat, un,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                yield out
                x_sample = out["sample"]

    def decode_first_stage(self, x_sample):
        # return nn.PixelShuffle(self.sf)(x_sample)
        return x_sample

    def encode_first_stage(self, y):
        return nn.PixelUnshuffle(self.sf)(y)

    def prior_sample(self, y, y_hat, un, noise=None):
        """
        Generate samples from the prior distribution, 
        i.e., q(x_T|x_0, y, y_hat) ~= N(x_T|y, kappa^2 * eta * I)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        if noise is None:
            noise = torch.randn_like(y)

        t = torch.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()
        return (y + _extract_into_tensor(self.sqrt_etas * self.kappa, t, y.shape) * un * noise)

    def training_losses(
            self, model, x_start, y, y_hat, un, t,
            # first_stage_model=None,
            model_kwargs=None,
            noise=None,
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param un: the uncertainty of y, which is the residual of LR and SR obtained by fidelity-oriented SR model
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        y = self.encode_first_stage(y)
        y_hat = self.encode_first_stage(y_hat)
        un = self.encode_first_stage(un)
        model_kwargs['un'] = un
        x_start = self.encode_first_stage(x_start)

        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            noise = self.encode_first_stage(noise)
        
        x_t = self.q_sample(x_start, y, y_hat, un, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            model_output = model(self._scale_input(x_t, un, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: x_start,
                ModelMeanType.RESIDUAL: y - x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape)*un,
            }[self.model_mean_type]
            assert target.shape == x_start.shape
            terms["mse"] = mean_flat((nn.PixelShuffle(self.sf)(target) - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["loss"] = terms["mse"] * weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, y=y, un=un, t=t, eps=model_output.detach())
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = self._predict_xstart_from_residual(y=y, residual=model_output.detach())
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = self._predict_xstart_from_eps_scale(x_t=x_t, y=y, un=un, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, self.decode_first_stage(x_t), self.decode_first_stage(pred_xstart)

    def _scale_input(self, inputs, un, t):
        if self.normalize_input:
            # var_un = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * (un.mean()**2 + un.std()**2) * self.kappa**2 + 0.5**2)
            var_un = 0.3
            std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 * var_un + 0.5**2)
            inputs_norm = inputs / std
        else:
            inputs_norm = inputs
        return inputs_norm

    def ddim_sample(
        self,
        model,
        x, y, y_hat, un, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        ddim_eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model=model,
            x_t=x,
            y=y,
            y_hat=y_hat,
            un=un, 
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        etas = _extract_into_tensor(self.etas, t, x.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x.shape)
        alpha = _extract_into_tensor(self.alpha, t, x.shape)
        sigma = ddim_eta * self.kappa * torch.sqrt(etas_prev / etas) * torch.sqrt(alpha)

        m_t = torch.sqrt(etas_prev / etas)

        k_t = (1 - etas_prev - (1 - etas) * m_t)

        y_t = (etas_prev - torch.sqrt(etas * etas_prev)) * y

        noise = torch.randn_like(x)
        mean_pred = (
            pred_xstart * k_t + x * m_t + y_t
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        y, y_hat, un,
        model,
        noise=None,
        noise_repeat=False,
        first_stage_model=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        one_step=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            y=y, y_hat=y_hat, un=un,
            model=model,
            noise=noise,
            noise_repeat=noise_repeat,
            first_stage_model=first_stage_model,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            ddim_eta=ddim_eta,
            one_step=one_step,
        ):
            final = sample["sample"]
        return nn.PixelShuffle(self.sf)(final)

    def ddim_sample_loop_progressive(
        self,
        y, y_hat, un,
        model,
        noise=None,
        noise_repeat=False,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        one_step=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device

        y = self.encode_first_stage(y)
        y_hat = self.encode_first_stage(y_hat)
        un = self.encode_first_stage(un)
        model_kwargs['un'] = un
        
        # generating noise
        if noise is None:
            noise = torch.randn_like(y)
        else:
            noise = self.encode_first_stage(noise)

        if noise_repeat:
            noise = noise[0,].repeat(y.shape[0], 1, 1, 1)

        x_sample = self.prior_sample(y, y_hat, un, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * y.shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model=model,
                    x=x_sample,
                    y=y, y_hat=y_hat, un=un,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    ddim_eta=ddim_eta,
                )
                if one_step:
                    out["sample"] = out["pred_xstart"]
                    yield out
                    break
                yield out
                x_sample = out["sample"]
