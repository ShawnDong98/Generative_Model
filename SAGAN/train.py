import os 
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from Attention import Generator, Discriminator
from utils import *

import argparse

def get_config():
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--version', type=str, default='sagan_1')
    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)
    # Path
    parser.add_argument('--image_path', type=str, default='/content/drive/My Drive/My Project/data1/')
    #parser.add_argument('--image_path', type=str, default='D:\Jupyter\GAN-zoo\data\own_data')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')
    # Step size
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)

    return parser.parse_args()

from dataloader import Data_Loader

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0)  # pause a bit so that plots are updated


class Trainer():
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.data_loader = Data_Loader(self.config.image_path, self.config.imsize, self.config.batch_size).loader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create directories if not exist
        make_folder(self.config.model_save_path, self.config.version)
        make_folder(self.config.sample_path, self.config.version)
        make_folder(self.config.attn_path, self.config.version)

        self.build_model()

        if self.config.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        self.G = Generator(self.config.batch_size, self.config.imsize, self.config.z_dim, self.config.g_conv_dim).to(self.device)
        self.D = Discriminator(self.config.batch_size, self.config.imsize, self.config.d_conv_dim).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.config.model_save_path, 'latest_G.pth')))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'latest_D.pth')))
        print('loaded trained models (step: {})..!'.format(self.configpretrained_model))
    

    def train(self):
        data_iter = iter(self.data_loader)

        fixed_z = torch.randn((self.config.batch_size, self.config.z_dim)).to(self.device)

        if self.config.pretrained_model:
            start = self.config.pretrained_model + 1
        else:
            start = 0

        start_time = time.time()

        for step in range(start, self.config.total_step):
            self.D.train()
            self.G.train()

            try:
                real_images, _ = next(data_iter)
                real_images = real_images.to(self.device)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)
                real_images = real_images.to(self.device)

            d_out_real, dr1, dr2 = self.D(real_images)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            z = torch.randn(real_images.size(0), self.config.z_dim).to(self.device)
            fake_images, gf1, gf2 = self.G(z)
            d_out_fake, dr1, dr2 = self.D(fake_images.detach())
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()


            z = torch.randn(real_images.size(0), self.config.z_dim).to(self.device)
            fake_images, _, _ = self.G(z)

            g_out_fake, _, _ = self.D(fake_images)
            g_loss_fake = -g_out_fake.mean()

            self.g_optimizer.zero_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            if (step + 1) % self.config.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.config.total_step, (step + 1),
                             self.config.total_step , d_loss_real,
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item()))


            if (step + 1) % self.config.sample_step == 0:
                fake_images,_,_= self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.config.sample_path, '{}_fake.png'.format(step + 1)))

            
            if (step + 1) % self.config.model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_G.pth'))
                torch.save(self.D.state_dict(),
                           os.path.join(self.config.model_save_path, 'latest_D.pth'))


trainer = Trainer()
trainer.train()