import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from celeba_hq import CELEB_A_HQ
from data_utils import preprocess



def jun_create(conf, dataset, mode='train'):
    data_path = conf['path']
    use_landmark = conf.get('use_landmark', False)
    conf = conf[mode]
    transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )
    
    if conf['name'] == 'celeba_hq':
        dataset = CELEB_A_HQ(dataset=dataset,
                                 mode=mode,
                                 transform=transformers,
                                 data_root=data_path,
                                 use_landmark=use_landmark
                                 )

    else:
        raise AttributeError(f'not support dataset config: {conf}')
    
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf['batch_size'],
        shuffle=True if mode == 'train' else False,
        pin_memory=True,
        drop_last=conf['drop_last'],
        num_workers=40,
    )

    return dl