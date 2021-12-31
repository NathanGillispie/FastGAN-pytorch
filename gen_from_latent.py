from typing import List
import re
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
import numpy as np
import os
import click
from tqdm import tqdm

from models import Generator

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)

def seed2vec(seed, batch_size, noise_dim):
    return np.random.RandomState(seed).randn(batch_size, noise_dim)

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

@click.command()
@click.pass_context
@click.option('--ckpt', help='path to checkpoint file')
@click.option('--save_dir', type=str, help='defaults to eval_imgs')
@click.option('--im_size', type=int, default=1024)
@click.option('--latent_vec', type=str, help='path to latent vector')
def generate_images(
    ctx: click.Context,
    ckpt: str,
    save_dir: str,
    im_size: int,
    latent_vec: str
):
    if latent_vec is None:
        dir = 'C:/Users/natha/repos/FastGAN-pytorch/train_results/nathanjack/models/'
        latent_vec = dir + os.listdir(dir)[-1]
        print('warning: latent_vec is defaulting to ' + latent_vec)
    if not os.path.exists(latent_vec):
        ctx.fail('latent_vec path does not exist')


    if ckpt is None:
        dir = 'C:/Users/natha/repos/FastGAN-pytorch/train_results/NathanGAN2/models/'
        ckpt = dir + os.listdir(dir)[-1]
        print('warning: ckpt is defaulting to ' + ckpt)
    if not os.path.exists(ckpt):
        ctx.fail('checkpoint path does not exist')

    if save_dir is None:
        save_dir = './eval_imgs_from_latent'
    os.makedirs(save_dir, exist_ok=True)

    noise_dim = 256
    device = torch.device('cuda')

    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=im_size)
    net_ig.to(device)

    checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
    # Remove prefix `module`.
    checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
    net_ig.load_state_dict(checkpoint['g'])
    load_params(net_ig, checkpoint['g_ema'])
    net_ig.eval()
    print('load checkpoint success')
    net_ig.to(device)

    del checkpoint

    dist = save_dir
    os.makedirs(dist, exist_ok=True)

    with torch.no_grad():
        z = torch.load(latent_vec, map_location=lambda a,b: a).to(device)
        g_imgs = net_ig(z)[0]
        g_imgs = F.interpolate(g_imgs, 512)
        for j, g_img in enumerate( g_imgs ):
            vutils.save_image(g_img.add(1).mul(0.5), 
                os.path.join(dist, f'latent_gen.png'))#, normalize=True, range=(-1,1))


if __name__ == "__main__":
    generate_images()