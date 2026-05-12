from basicsr.utils import FileClient, imfrombytes, img2tensor, imwrite
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.data.data_util import paths_from_lmdb, scandir

file_client = FileClient('disk')
gt_folder = '/home/datasets/imagenet256/gt'
paths = sorted(list(scandir(gt_folder, full_path=True)))

for path in paths:
    img_bytes = file_client.get(path, 'gt')
    img_gt = imfrombytes(img_bytes, float32=True)
    img_lq = imresize(img_gt, 0.25)
    lq_path = path.replace('gt', 'lq_bicubic_matlab')
    # print(img_lq*255)
    imwrite(img_lq*255, lq_path)
    print(lq_path)
