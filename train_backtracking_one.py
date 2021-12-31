import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import argparse
from tqdm import tqdm
import click
import os
from models import Generator
from operation import load_params
from operation import ImageFolder, InfiniteSamplerWrapper
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

#torch.backends.cudnn.benchmark = True

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        #pred, [rec_all, rec_small, rec_part], part = net(data, label)
        pred = net(data, label)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() #+ \
            #percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            #percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            #percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item()#, rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
@torch.no_grad()
def interpolate(z1, z2, netG, img_name, step=8):
    z = [  a*z2 + (1-a)*z1 for a in torch.linspace(0, 1, steps=step)  ]
    z = torch.cat(z).view(step, -1)
    g_image = netG(z)[0]
    vutils.save_image( g_image.add(1).mul(0.5), img_name , nrow=step)

@click.command()
@click.pass_context
@click.option('--ckpt', default='./train_results/NathanGAN2/models/all_22600.pth', help='path to checkpoint file')
@click.option('--name', default='nathanjack', help='name of experiment')
@click.option('--path', type=str, default='./unseen_images/', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
@click.option('--iter', type=int, default=5000, help='number of iterations')
@click.option('--batch_size', type=int, default=8, help='number of images in minibatch')
@click.option('--im_size', type=int, default=1024, help='image size')
def train(
    ctx: click.Context,
    ckpt: str,
    name: str,
    path: str,
    iter: int,
    batch_size: int,
    im_size: int
):
    total_iterations = iter
    ngf = 64
    nz = 256
    nbeta1 = 0.5
    use_cuda = False
    dataloader_workers = 4
    current_iteration = 0
    save_interval = 20

    if not os.path.exists(ckpt):
        ctx.fail('checkpoint path does not exist')

    saved_model_folder = os.path.join('train_results' , name , 'models')
    saved_image_folder = os.path.join('train_results' , name , 'images')
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    dataset = ImageFolder(root=path, transform=trans)
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)

    ckpt = torch.load(ckpt)
    load_params( netG , ckpt['g_ema'] )
    #netG.eval()
    netG.to(device)

    fixed_noise = torch.randn(batch_size, nz, requires_grad=True, device=device)
    optimizerG = optim.Adam([fixed_noise], lr=0.1, betas=(nbeta1, 0.999))
    
    real_image = next(dataloader).to(device)

    log_rec_loss = 0

    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        optimizerG.zero_grad()
        g_image = netG(fixed_noise)[0]
        rec_loss = percept( F.avg_pool2d( g_image, 2, 2), F.avg_pool2d(real_image,2,2) ).sum() + 0.2*F.mse_loss(g_image, real_image)
        rec_loss.backward()
        optimizerG.step()
        log_rec_loss += rec_loss.item()

        if iteration % 100 == 0:
            print("lpips loss g: %.5f"%(log_rec_loss/100))
            log_rec_loss = 0

        if iteration % (save_interval*2) == 0:
            with torch.no_grad():
                vutils.save_image( torch.cat([
                        real_image, g_image]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
        # interpolate(fixed_noise[0], fixed_noise[1], netG, saved_image_folder+'/interpolate_0_1_%d.jpg'%iteration)
        
        if iteration % (save_interval*5) == 0 or iteration == total_iterations:
            torch.save(fixed_noise, saved_model_folder+'/%d.pth'%iteration)

if __name__ == "__main__":
    train()