from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_psnr_pt, calculate_ssim_pt
from .lpips import calculate_lpips_pt, calculate_lpips
from .pyiqa_metrics import calculate_pyiqa_metric_pt

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_psnr_pt', 'calculate_ssim_pt', 'calculate_lpips_pt', 'calculate_lpips',
           'calculate_pyiqa_metric_pt']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
