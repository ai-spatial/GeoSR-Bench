import cv2
import numpy as np
import torch
import torch.nn.functional as F
import lpips

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.color_util import rgb2ycbcr_pt
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils.img_util import img2tensor, tensor2img


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, lpips_loss=None, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (np.ndarray): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (np.ndarray): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        lpips_loss (nn.Module): LPIPS model. Default: lpips.LPIPS(net='vgg').
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: LPIPS result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    if lpips_loss is None:
        lpips_loss = lpips.LPIPS(net='vgg')
        lpips_loss.eval()
        lpips_loss = torch.compile(lpips_loss, mode='reduce-overhead')
    img = img2tensor(img).unsqueeze(0).cuda() / 255
    img2 = img2tensor(img2).unsqueeze(0).cuda() / 255
    # print(img, img2)
    lpips_loss.to(img.device)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)

    lpips_out = lpips_loss.forward(img, img2).reshape(-1, 1)
    return lpips_out.item()


@METRIC_REGISTRY.register()
def calculate_lpips_pt(img, img2, crop_border, lpips_loss=None, input_order='HWC', test_y_channel=False, **kwargs):
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
    # print(img.device, type(lpips_loss))
    assert lpips_loss is not None
    # if lpips_loss is None:
    #     lpips_loss = lpips.LPIPS(net='vgg')
    #     lpips_loss.eval()
    #     lpips_loss = torch.compile(lpips_loss, mode='reduce-overhead')
    lpips_loss.to(img.device)

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float32)
    img2 = img2.to(torch.float32)

    lpips_out = lpips_loss(img, img2).reshape(-1, 1)
    return lpips_out