import random
import paddle
import paddle.nn as nn
from paddle.vision.transforms import InterpolationMode, Resize, RandomCrop

class CustomRandomResize(nn.Layer):

    def __init__(self, scale=(0.5, 2.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.min_scale, self.max_scale = min(scale), max(scale)
        self.interpolation = interpolation

    def forward(self, img):
        if isinstance(img, paddle.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = [int(height * scale), int(width * scale)]
        img = Resize(new_size, interpolation=self.interpolation)(img)

        return img


class CustomRandomCrop(RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed, fill=fill, padding_mode=padding_mode)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        width, height = img.shape[-2], img.shape[-1]

        tar_h, tar_w = self.size
        tar_h = min(tar_h, height)
        tar_w = min(tar_w, width)
        i, j, h, w = self.get_params(img, (tar_h, tar_w))

        return F.crop(img, i, j, h, w)
