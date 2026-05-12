import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyiqa

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils.img_util import img2tensor, tensor2img

@METRIC_REGISTRY.register()
def calculate_pyiqa_metric_pt(img, img2=None, crop_border=0, metric=None, test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        lpips_loss (nn.Module): LPIPS model. Default: lpips.LPIPS(net='vgg').
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: LPIPS result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    assert metric is not None
    metric.to(img.device)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        if img2 is not None:
            img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        if img2 is not None:
            img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float32)
    if img2 is not None:
        img2 = img2.to(torch.float32)

    metric_out = metric(img, img2) if img2 is not None else metric(img)
    return metric_out
