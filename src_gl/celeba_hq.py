import torch
from torch import Tensor
from typing import List, Tuple, Any, Optional
from torchvision.transforms import functional as F
import torchvision

import torch.utils.data as torch_data
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import math
import os
import numpy as np


class CELEB_A_HQ(torch_data.Dataset):
    def __init__(self,
                 dataset,
                 mode="train",
                 transform=transforms.ToTensor(),
                 data_root='',
                 use_landmark=False,
                 local_rank=0):
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''
        self.dataset = dataset
        self.mode = mode
        self.transform = transform
        self.data_root = data_root
        self.use_landmark = True
        self.local_rank = f'cuda:{local_rank}'
        self.fa = None

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        source_image = np.array(Image.open(os.path.join(self.data_root, self.dataset.iloc[[index]]['path'].values[0])))
        target_image = deepcopy(source_image)
        masked = np.zeros_like(target_image)[:, :, 0]
        if self.mode == 'train':
            mid = int((self.dataset.iloc[[index]]['28'].values[0] + self.dataset.iloc[[index]]['4'].values[0]) / 2)

            width = int(abs(self.dataset.iloc[[index]]['28'].values[0] - self.dataset.iloc[[index]]['4'].values[0]))
            height = width * 0.77
            left_x = int((self.dataset.iloc[[index]]['56'].values[0] + mid) / 2 - width // 2)
            left_y = int(self.dataset.iloc[[index]]['57'].values[0])

            bbx1, bby1, bbx2, bby2 = self.randbbox(width, height, lam=0)

            bbx1 = int(bbx1 + left_x)
            bbx2 = int(bbx2 + left_x)
            bby1 = int(bby1 + left_y)
            bby2 = int(bby2 + left_y)

            if bbx1 > bbx2:
                bbx1, bbx2 = bbx2, bbx1
            if bby1 > bby2:
                bby1, bby2 = bby2, bby1
            
            source_image[bby1:bby2, bbx1:bbx2] = 128
            masked[bby1:bby2, bbx1:bbx2] = 255
            # np.random.randint(low=0, high=255, size=source_image[bby1:bby2, bbx1:bbx2].shape)
        else:
            mid = int((self.dataset.iloc[[index]]['28'].values[0] + self.dataset.iloc[[index]]['4'].values[0]) / 2)

            width = abs(self.dataset.iloc[[index]]['28'].values[0] - self.dataset.iloc[[index]]['4'].values[0]) * 0.95
            height = width * 0.78
            left_x = int((self.dataset.iloc[[index]]['56'].values[0] + mid) / 2 - width // 2)
            left_y = int(self.dataset.iloc[[index]]['57'].values[0])
            
            source_image[left_y:int(left_y + height), left_x:int(left_x + width)] = 128
            masked[left_y:int(left_y + height), left_x:int(left_x + width)] = 255
            # p.random.randint(low=0, high=255, size=source_image[left_y:int(left_y + height), left_x:int(left_x + width)].shape)

        landmark = []
        for i in range(68 * 2):
            landmark.append(self.dataset.iloc[[index]][f'{i}'].values[0])

        source_image = Image.fromarray(source_image)
        target_image = Image.fromarray(target_image)
        masked = Image.fromarray(masked)
        # if self.use_landmark is not None:
            # aug_channel = Image.fromarray(aug_channel)
        if self.mode == 'train':
            if np.random.rand() < 0.5:
                source_image = F.hflip(source_image)
                target_image = F.hflip(target_image)
                # if self.use_landmark is not None:
                    # aug_channel = F.hflip(aug_channel)

            i, j, h, w = self.get_params(source_image, (0.75, 1.), (3. / 4., 4. / 3.))
            source_image = F.resized_crop(source_image, i, j, h, w, (256, 256))
            target_image = F.resized_crop(target_image, i, j, h, w, (256, 256))
            masked = F.resized_crop(masked, i, j, h, w, (256, 256))
            # if self.use_landmark is not None:
                # aug_channel = F.resized_crop(aug_channel, i, j, h, w, (256, 256))
        source = self.transform(source_image)
        target = self.transform(target_image)
        masked = torchvision.transforms.ToTensor()(masked)
        # if self.use_landmark is not None:
        #     aug_channel = self.transform(aug_channel)
        #     source = torch.cat([source, aug_channel[:1, :, :]], dim=0)

        return source, target, np.array(landmark, dtype=np.float32), masked

    def randbbox(self, width, height, lam):
        W = width
        H = height

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        #cx = np.random.randint(W)
        #cy = np.random.randint(H)
        alpha = 80.0
        beta = 80.0
        cx = int(W * np.random.beta(alpha, beta))
        cy = int(H * np.random.beta(alpha, beta))

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def get_params(self,
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w