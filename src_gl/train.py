from __future__ import print_function
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from math import log10

from datetime import datetime
from tqdm import tqdm
import logging
import logging.handlers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from net_gl import _netlocalD,_netG

from TensorData_Loader import get_data_set
from tensorboardX import SummaryWriter

from IQA_pytorch import SSIM

from jun_dataset import jun_create
import argparse
import wandb
import face_alignment
import timm

DATAPATH='/opt/project/data1024x1024'
DATA_CONFIG = {
    "path": '/opt/project/data1024x1024',
    "train": {
        'name': 'celeba_hq',
        'batch_size': 64,
        'drop_last': True,
        'use_landmark': False,
  
        'preprocess': 
            [
                {
                    'type': 'tensor'
                },
                {
                    'type': 'normalize',
                    'params': {
                        'mean': 0.5,
                        'std': 0.5
                    }
                }
            ]
        
    },
    'test': {
        'name': 'celeba_hq',
        'batch_size': 1,
        'drop_last': False,
        'use_landmark': False,
  
        'preprocess': 
            [
                {
                    'type': 'tensor'
                },
                {
                    'type': 'resize',
                    'params': {
                        'size': 256
                    }
                },
                {
                    'type': 'normalize',
                    'params': {
                        'mean': 0.5,
                        'std': 0.5
                    }
                }
            ]
        
    }
}

from typing import Tuple, Optional, List
import math
import torch.nn as nn
import torch.nn.functional as F

def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1), value: float = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# Training settings
parser = argparse.ArgumentParser(description='Globally and Locally Consistent Image Completion')
parser.add_argument('--label', default='compare_gl' , help='facades')
parser.add_argument('--datasetPath', default='../dataset/Facade' , help='facades')

parser.add_argument('--resume_epoch', default=0 , help='facades')
parser.add_argument('--nEpochs', type=int, default=3*1000, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nef', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')

parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
parser.add_argument('--gan_wei', type=int, default=0.0004, help='weight on L1 term in objective')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--G_model', type=str, default="gl")
parser.add_argument('--D_model', type=str, default="gl_d")

#-----Sett logging--------
logger = logging.getLogger('mylogger')
fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setFormatter(fomatter)
logger.addHandler(streamHandler)
logger.setLevel(logging.INFO)

def train(epoch, detector=None, fa=None):
    real_label = 1
    fake_label = 0
    netG.train()
    netD.train()
    for iteration, (input_masked, target, _, masked) in tqdm(enumerate(training_data_loader, 1), total=len(training_data_loader)):
        input_masked = input_masked.to(device)
        target = target.to(device)
        mask_tensor = target.to(device)

        if fa is not None:
            aug_source = torch.zeros((input_masked.shape[0], 4, input_masked.shape[2], input_masked.shape[3])).to(device)
            preds_noise = fa.get_landmarks_from_batch(input_masked * 255)
            aug_channel = torch.zeros((input_masked.shape[0], 1, input_masked.shape[2], input_masked.shape[3])).to(device)
            for idx, prediction in enumerate(preds_noise):
                if prediction is not None:
                    for x, y in prediction:
                        if x >= aug_channel.shape[0]:
                            x = aug_channel.shape[0] - 1
                        if y >= aug_channel.shape[1]:
                            y = aug_channel.shape[1] - 1

                        aug_channel[idx, 0, int(x)][int(y)] = 0.5
            aug_source = torch.cat([input_masked, aug_channel], dim=1)
            input_masked = aug_source
        if detector is not None:
            aug_source = torch.zeros((input_masked.shape[0], 4, input_masked.shape[2], input_masked.shape[3])).to(device)
            preds_noise = detector(input_masked)
            aug_channel = torch.zeros((input_masked.shape[0], 1, input_masked.shape[2], input_masked.shape[3])).to(device)

            for idx, prediction in enumerate(preds_noise):
                for j in range(len(prediction) // 2):
                    x = prediction[j * 2]
                    y = prediction[j * 2 + 1]
                    if x >= aug_channel.shape[0]:
                        x = aug_channel.shape[0] - 1
                    if y >= aug_channel.shape[1]:
                        y = aug_channel.shape[1] - 1

                    aug_channel[idx, 0, int(x)][int(y)] = 0.5
            aug_source = torch.cat([input_masked[:, :3], aug_channel], dim=1)
            input_masked = aug_source

        # print(masked)
        # mask_tensor = masked.unsqueeze(1)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        netD.zero_grad()
        real_cpu = target.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device).unsqueeze(-1)
        label = label.float()
        output = netD(real_cpu, mask_tensor[:, :3])
        D_real_loss = criterionGAN(output, label)
        D_real_loss.backward()
        D_x = output.mean().item()

        # train with fake
        fake = netG(input_masked)
        label.fill_(fake_label)
        output = netD(fake.detach(), mask_tensor[:, :3])
        D_fake_loss = criterionGAN(output, label)
        D_fake_loss.backward()
        D_G_z1 = output.mean().item()
        loss_d = (D_real_loss + D_fake_loss) * opt.gan_wei
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        pred_fake = netD(fake, mask_tensor)

        loss_g_gan = criterionGAN(pred_fake, label)
        loss_g_l2 = criterionMSE(fake, target)
        loss_g = opt.gan_wei * loss_g_gan + loss_g_l2

        loss_g.backward()
        optimizerG.step()

    wandb.log({
        'train/d_loss': loss_d.item(),
        'train/D_real_loss': D_real_loss.item(),
        'train/D_fake_loss': D_fake_loss.item(),
        'train/loss_g': loss_g.item(),
        'train/loss_g_gan': loss_g_gan.item(),
        'train/loss_g_l2': loss_g_l2.item()
    }, step=epoch)
    # writer.add_scalar('dataD/d_loss', loss_d.item(), epoch)
    # writer.add_scalar('dataD/D_real_loss', D_real_loss.item(), epoch)
    # writer.add_scalar('dataD/D_fake_loss', D_fake_loss.item(), epoch)

    # writer.add_scalar('dataG/loss_g', loss_g.item(), epoch)
    # writer.add_scalar('dataG/loss_g_gan', loss_g_gan.item(), epoch)
    # writer.add_scalar('dataG/loss_g_l2', loss_g_l2.item(), epoch)

def ToPilImage(image_tensor):
    image_norm = (image_tensor.data + 1 )/2
    return  image_norm


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def test(epoch, detector=None, fa=None):
    sum_psnr = 0
    prediction = 0
    netG.eval()
    metric = SSIM(channels=3)
    sum_ssim = 0
    for (input_masked, target, _, masked) in testing_data_loader:
        input_masked = input_masked.to(device)
        target = target.to(device)

        if fa is not None:
            aug_source = torch.zeros((input_masked.shape[0], 4, input_masked.shape[2], input_masked.shape[3])).to(device)
            preds_noise = fa.get_landmarks_from_batch(input_masked * 255)
            aug_channel = torch.zeros((input_masked.shape[0], 1, input_masked.shape[2], input_masked.shape[3])).to(device)
            for idx, prediction in enumerate(preds_noise):
                if prediction is not None:
                    for x, y in prediction:
                        if x >= aug_channel.shape[0]:
                            x = aug_channel.shape[0] - 1
                        if y >= aug_channel.shape[1]:
                            y = aug_channel.shape[1] - 1

                        aug_channel[idx, 0, int(x)][int(y)] = 0.5
            aug_source = torch.cat([input_masked, aug_channel], dim=1)
            input_masked = aug_source
        if detector is not None:
            aug_source = torch.zeros((input_masked.shape[0], 4, input_masked.shape[2], input_masked.shape[3])).to(device)
            preds_noise = detector(input_masked)
            aug_channel = torch.zeros((input_masked.shape[0], 1, input_masked.shape[2], input_masked.shape[3])).to(device)

            for idx, prediction in enumerate(preds_noise):
                for j in range(len(prediction) // 2):
                    x = prediction[j * 2]
                    y = prediction[j * 2 + 1]
                    if x >= aug_channel.shape[0]:
                        x = aug_channel.shape[0] - 1
                    if y >= aug_channel.shape[1]:
                        y = aug_channel.shape[1] - 1

                    aug_channel[idx, 0, int(x)][int(y)] = 0.5
            aug_source = torch.cat([input_masked[:, :3], aug_channel], dim=1)
            input_masked = aug_source
        # input, target, input_masked = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        prediction = netG(input_masked)
        ssim = metric(target, prediction, as_loss=False).item()
        sum_ssim += ssim
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        sum_psnr += psnr

    avg_psnr = sum_psnr / len(testing_data_loader)
    avg_ssim = sum_ssim / len(testing_data_loader)
    logger.info("[{}]===> Avg. PSNR: {:.4f} dB".format(epoch, avg_psnr))
    logger.info("[{}]===> Avg. SSIM: {:.4f} dB".format(epoch, avg_ssim))

    wandb.log({
        'test/PSNR': avg_psnr,
        'test/SSIM': avg_ssim,
        'test/input': wandb.Image(ToPilImage(input_masked[0])),
        'test/prediction': wandb.Image(ToPilImage(prediction[0])),
        'test/target': wandb.Image(ToPilImage(target[0]))
    }, step=epoch)
    # writer.add_image('Test/input', ToPilImage(input_masked[0]), epoch)
    # writer.add_image('Test/prediction', ToPilImage(prediction[0]), epoch)
    # writer.add_image('Test/target', ToPilImage(target[0]), epoch)
    # writer.add_scalar('PSNR/PSNR', avg_psnr, epoch)
    # writer.add_scalar('SSIM/SSIM', avg_ssim, epoch)

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint/{}_{}".format(opt.label,opt.G_model))) :
        os.mkdir(os.path.join("checkpoint/{}_{}".format(opt.label,opt.G_model)))
    net_g_model_out_path = "checkpoint/{}_{}/netG_model_epoch_{}.pth".format(opt.label,opt.G_model, epoch)
    net_d_model_out_path = "checkpoint/{}_{}/netD_model_epoch_{}.pth".format(opt.label,opt.G_model, epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)

    logger.info("Checkpoint saved to {}".format("checkpoint" + opt.label))

if __name__ ==  '__main__':
    wandb.init(project='GLCM')
    wandb.run.name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    opt = parser.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")

    logger.info('===> Loading datasets')

    data = pd.read_csv(os.path.join(DATAPATH, 'landmark.csv'))
        
    train_data, test_data = train_test_split(data, test_size=0.025, random_state=42)
    train_data.reset_index(drop=True, inplace=True)
    train_data = train_data.rename_axis('index').reset_index()
    test_data.reset_index(drop=True, inplace=True)
    test_data = test_data.rename_axis('index').reset_index()

    training_data_loader = jun_create(
        DATA_CONFIG,
        dataset=train_data,
        mode='train'
    )
    testing_data_loader = jun_create(
        DATA_CONFIG,
        dataset=test_data,
        mode='test'
    )

    # train_set = get_data_set(opt.datasetPath, "train")
    # test_set = get_data_set(opt.datasetPath, "test")
    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
    #                                   shuffle=True)
    # testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
    #                                  shuffle=True)


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    logger.info('===> Building model')
    resume_epoch = 0
    if opt.resume_epoch > 0:
        resume_epoch = opt.resume_epoch
        net_g_model_out_path = "checkpoint/{}_{}/netG_landmark_model_epoch_{}.pth".format(opt.label, opt.G_model, resume_epoch)
        net_d_model_out_path = "checkpoint/{}_{}/netD_landmark_model_epoch_{}.pth".format(opt.label, opt.G_model, resume_epoch)
        netG = torch.load(net_g_model_out_path)
        netD = torch.load(net_d_model_out_path)
    else:
        netG = _netG(opt)
        netG.apply(weights_init)
        netD = _netlocalD(opt.nc, opt.ndf, n_layers=3)
        netD.apply(weights_init)

    criterionGAN = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    # setup optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    logger.info('---------- Networks initialized -------------')
    print_network(netG)
    print_network(netD)
    logger.info('-----------------------------------------------')

    netD = netD.to(device)
    netG = netG.to(device)

    netD = torch.nn.DataParallel(netD, device_ids=[0,1,2,3])
    netG = torch.nn.DataParallel(netG, device_ids=[0,1,2,3])

    criterionGAN = criterionGAN.to(device)
    criterionMSE = criterionMSE.to(device)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs_'+opt.label, opt.label + "_" + str(opt.resume_epoch) + "_" + current_time)

    if True:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
        detector  = timm.create_model('tf_efficientnet_b1_ns', pretrained=False, num_classes=68 * 2)
        detector.conv_stem = Conv2dSame(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        data = torch.load('/opt/project/face_reconstruction/checkpoints/efficient_landmark/best_80.chpt')
        detector.load_state_dict({k.replace('module.', ''): v for k, v in data['detector'].items()})
        detector.to(device)
        detector = torch.nn.DataParallel(detector, device_ids=[0,1,2,3])
        detector.eval()
    else:
        fa = None
        detector = None

    logger.info('Train Start')
    for epoch in range(resume_epoch, opt.nEpochs + 1):
        train(epoch, detector, fa)
        test(epoch, detector, fa)
        if epoch % 10 == 0:        checkpoint(epoch)

    wandb.finish()


