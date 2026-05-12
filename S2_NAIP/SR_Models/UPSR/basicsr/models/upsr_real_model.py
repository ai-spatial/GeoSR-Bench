import torch
import torch.amp as amp
import torch.nn as nn
import functools
import math
import lpips
import os
import os.path as osp
import pyiqa
import numpy as np
import random
import tqdm
import torchvision
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import matplotlib.pyplot as plt


# from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_obj_from_str, get_root_logger, ImageSpliterTh, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from .base_model import BaseModel
from .sr_model import SRModel
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from contextlib import nullcontext
from copy import deepcopy
from torch.nn.parallel import DataParallel
from collections import OrderedDict
from torchvision import transforms
from PIL import Image


class UPSRRealModel(SRModel):
    """Diffusion SR model for single image super-resolution."""

    def __init__(self, opt):
        super(UPSRRealModel, self).__init__(opt)
        self.opt = opt

        logger = get_root_logger()

        self.sf = self.opt['scale']

        # define network net_mse g(\cdot)
        net_mse_opt = self.opt['network_mse']
        assert net_mse_opt['ckpt']['path'] is not None, 'ckpt_path is required for net_mse'
        logger.info(f"Restoring network_mse from {net_mse_opt['ckpt']['path']}")

        self.net_mse = build_network(net_mse_opt)
        param_key = net_mse_opt['ckpt'].get('param_key_mse', 'params_ema')
        self.load_network(self.net_mse, net_mse_opt['ckpt']['path'], net_mse_opt['ckpt'].get('strict_load_mse', True), param_key)
        self.net_mse.eval()
        for name, param in self.net_mse.named_parameters():
            param.requires_grad = False
        self.net_mse = self.net_mse.to(self.device)

        # define base_diffusion
        diff_opt = self.opt['diffusion']
        self.base_diffusion = build_network(diff_opt)
        
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 160)

        # define lpips loss
        loss_lpips = self.metric_lpips = pyiqa.create_metric('lpips-vgg', as_loss=True, device=self.device)
        self.loss_lpips = loss_lpips

        if self.opt['rank'] == 0:
            self.metrics_fr = {}
            self.metrics_nr = {}

            for metric_name, metric_opt in self.opt['val']['metrics'].items():
                if metric_opt.get('fr', True):
                    self.metrics_fr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)
                else:
                    self.metrics_nr[metric_name] = pyiqa.create_metric(metric_name, device=self.device)

            self.metrics_fr['psnr'] = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=self.device)
            self.metrics_fr['ssim'] = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr', device=self.device)


    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            if param_key in load_net:
                load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):

        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = True
            self.perceptual_weight = train_opt['perceptual_opt']['lpips_weight']
        else:
            self.cri_perceptual = False

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.amp_scaler = amp.GradScaler() if self.opt['train'].get('use_fp16', False) else None

    def backward_step(self, dif_loss_wrapper, micro_lq, micro_gt, num_grad_accumulate, tt):
        loss_dict = OrderedDict()

        context = amp.autocast if self.opt['train'].get('use_fp16', False) else nullcontext
        with context(device_type="cuda"):
            losses, x_t, x0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            l_pix = losses['loss'].mean() / num_grad_accumulate
            
            l_total = l_pix
            loss_dict['l_pix'] = l_pix 

            if self.cri_perceptual:
                l_lpips = self.loss_lpips(x0_pred[:, :3, :, :].clamp(-1., 1.), micro_gt[:, :3, :, :]).to(x0_pred.dtype).view(-1)
                if torch.any(torch.isnan(l_lpips)):
                    l_lpips = torch.nan_to_num(l_lpips, nan=0.0)
                l_lpips = l_lpips.mean() / num_grad_accumulate * self.perceptual_weight

                l_total += l_lpips
                loss_dict['l_lpips'] = l_lpips 

        if self.amp_scaler is None:
            l_total.backward()
        else:
            self.amp_scaler.scale(l_total).backward()

        return loss_dict, x_t, x0_pred

    def optimize_parameters(self, current_iter):  
        current_batchsize = self.lq.shape[0]
        micro_batchsize = self.opt['datasets']['train']['micro_batchsize']
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        self.optimizer_g.zero_grad()

        loss_dict = OrderedDict()
        loss_dict['l_pix'] = 0
        if self.cri_perceptual:
            loss_dict['l_lpips'] = 0

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_lq = self.lq[jj:jj+micro_batchsize,]
            micro_gt = self.gt[jj:jj+micro_batchsize,]


            last_batch = (jj+micro_batchsize >= current_batchsize)
            if self.opt['diffusion'].get('one_step', False):
                tt = torch.ones(
                    size=(micro_gt.shape[0],),
                    device=self.lq.device,
                    dtype=torch.int32,
                    ) * (self.base_diffusion.num_timesteps - 1)
            else:
                tt = torch.randint(
                        0, self.base_diffusion.num_timesteps,
                        size=(micro_gt.shape[0],),
                        device=self.lq.device,
                        )
            
            with torch.no_grad():        
                # y_0
                micro_lq_bicubic = torch.nn.functional.interpolate(
                        micro_lq, scale_factor=self.sf, mode='bicubic', align_corners=False,
                        )
                # g(y_0)
                micro_sr_mse = (self.net_mse(micro_lq * 0.5 + 0.5) - 0.5) / 0.5

                # un
                if self.opt['diffusion']['un'] > 0:
                    diff = (micro_sr_mse - micro_lq_bicubic) / 2
                    un_max = self.opt['diffusion']['un']
                    b_un = self.opt['diffusion']['min_noise']
                    micro_uncertainty = torch.abs(diff).clamp_(0., un_max) / un_max
                    micro_uncertainty = b_un + (1 - b_un) * micro_uncertainty
                else:
                    micro_uncertainty = torch.ones_like(micro_sr_mse)

            # n
            noise = torch.randn_like(micro_sr_mse)

            lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([micro_sr_mse, micro_lq_bicubic], dim=1))


            model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.net_g,
                micro_gt,
                micro_lq_bicubic,
                micro_sr_mse,
                micro_uncertainty,
                tt,
                model_kwargs=model_kwargs,
                noise=noise,
            )

            if last_batch or self.opt['num_gpu'] <= 1:
                losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
            else:
                with self.net_g.no_sync():
                    losses, x_t, x0_pred = self.backward_step(compute_losses, micro_lq, micro_gt, num_grad_accumulate, tt)
            
            loss_dict['l_pix'] += losses['l_pix']
            if self.cri_perceptual:
                loss_dict['l_lpips'] += losses['l_lpips']
            
        if self.opt['train'].get('use_fp16', False):
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()
        else:
            self.optimizer_g.step()

        self.net_g.zero_grad()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def sample_func(self, y0, noise_repeat=False):
        desired_min_size = self.opt['val']['desired_min_size']
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        y_bicubic = torch.nn.functional.interpolate(
            y0, scale_factor=self.sf, mode='bicubic', align_corners=False,
            )
        
        y_hat = (self.net_mse(y0 * 0.5 + 0.5) - 0.5) / 0.5
        if self.opt['diffusion']['un'] > 0:
            diff = (y_hat - y_bicubic) / 2
            un_max = self.opt['diffusion']['un']
            b_un = self.opt['diffusion']['min_noise']
            un = torch.abs(diff).clamp_(0., un_max) / un_max
            un = b_un + (1 - b_un) * un
        else:
            un = torch.ones_like(y_hat)

        lq_cond = nn.PixelUnshuffle(self.sf)(torch.cat([y_hat, y_bicubic], dim=1))

        model_kwargs={'lq':lq_cond,} if self.opt['network_g']['params']['cond_lq'] else None
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.net_g_ema
        else:
            self.net_g.eval()
            net = self.net_g
        results = self.base_diffusion.ddim_sample_loop(
                y=y_bicubic,
                y_hat=y_hat,
                un=un,
                model=net,
                first_stage_model=None,
                noise=None,
                noise_repeat=noise_repeat,
                # clip_denoised=(self.autoencoder is None),
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=self.opt['diffusion'].get('one_step', False),
                )    

        if flag_pad:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

        return results.clamp_(-1.0, 1.0)

    def test(self):

        def _process_per_image(im_lq_tensor):
            if im_lq_tensor.shape[2] > self.opt['val']['chop_size'] or im_lq_tensor.shape[3] > self.opt['val']['chop_size']:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.opt['val']['chop_size'],
                        stride=self.opt['val']['chop_stride'],
                        sf=self.opt['scale'],
                        extra_bs=self.opt['val']['chop_bs'],
                        )
                for im_lq_pch, index_infos in im_spliter:
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            noise_repeat=self.opt['val']['noise_repeat'],
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        noise_repeat=self.opt['val']['noise_repeat'],
                        )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor
        
        self.output = _process_per_image(self.lq)


    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data, training=True):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if training and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM sharpen the GT images
            if self.opt['degradation']['use_sharp'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['degradation']['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['degradation']['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['degradation']['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['degradation']['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['degradation']['scale'] * scale), int(ori_w / self.opt['degradation']['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['degradation']['gray_noise_prob2']
            if np.random.uniform() < self.opt['degradation']['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['degradation']['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['degradation']['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['degradation']['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['degradation']['scale'], ori_w // self.opt['degradation']['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['degradation']['gt_size']
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['degradation']['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
            # normalize
            # if self.mean is not None or self.std is not None:
            self.lq = (self.lq - 0.5) / 0.5
            self.gt = (self.gt - 0.5) / 0.5
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
            else:
                self.gt = None

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm.tqdm(total=len(dataloader), unit='image')

        num_img = 0

        for idx, val_data in enumerate(dataloader):
            num_img += len(val_data['lq_path'])
            self.feed_data(val_data, training=False)
            
            self.test()

            metric_data['img'] = self.output.clamp(0, 1)
            # metric_data['img'] = torch.clamp((self.output * 255.0).round(), 0, 255) / 255.
            metric_data['img2'] = self.gt

            if with_metrics:
                # calculate metrics
                if metric_data['img2'] is not None:
                    for name, metric in self.metrics_fr.items():
                        self.metric_results[name] += metric(metric_data['img'], metric_data['img2']).sum().item()
                for name, metric in self.metrics_nr.items():
                    self.metric_results[name] += metric(metric_data['img']).sum().item()

            visuals = self.get_current_visuals()

            sr_img = [tensor2img(visuals['result'][ii]) for ii in range(self.output.shape[0])]
            if 'gt' in visuals:
                # gt_img = tensor2img([visuals['gt']])
                # metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output

            torch.cuda.empty_cache()

            
            for ii in range(len(val_data['lq_path'])):
                if save_img:
                    img_name = osp.splitext(osp.basename(val_data['lq_path'][ii]))[0]

                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}_{self.opt["name"]}.png')
                            
                    imwrite(sr_img[ii], save_img_path)
                    
            
            if use_pbar:
                pbar.update(1)

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= num_img
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt') and self.gt is not None:
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
