import os
import torch 
from torchvision import utils as vutils
import cv2
from tqdm import tqdm
import numpy as np
import click
#import pandas as pd

from easing_functions.easing import LinearInOut
from easing_functions import QuadEaseInOut
from easing_functions import SineEaseIn, SineEaseInOut, SineEaseOut
from easing_functions import ElasticEaseIn, ElasticEaseInOut, ElasticEaseOut

ease_fn_dict = {'QuadEaseInOut': QuadEaseInOut,
                'SineEaseIn': SineEaseIn, 
                'SineEaseInOut': SineEaseInOut, 
                'SineEaseOut': SineEaseOut,
                'ElasticEaseIn': ElasticEaseIn, 
                'ElasticEaseInOut': ElasticEaseInOut, 
                'ElasticEaseOut': ElasticEaseOut,
                'Linear': LinearInOut}

def interpolate(z1, z2, num_interp):
    # this is a "first frame included, last frame excluded" interpolation
    w = torch.linspace(0, 1, num_interp+1)
    interp_zs = []
    for n in range(num_interp):
        interp_zs.append( (z2*w[n].item() + z1*(1-w[n].item())).unsqueeze(0) )
    return torch.cat(interp_zs)

def seed2vec(seed, G_z_dim = 512):
  return np.random.RandomState(seed).randn(1, G_z_dim)

def interpolate_ease_inout(z1, z2, num_interp, ease_fn, model_type='freeform'):
    # this is a "first frame included, last frame excluded" interpolation
    w = ease_fn(start=0, end=1, duration=num_interp+1)
    interp_zs = []

    # just to make sure the latent vector's in the right shape
    if model_type == 'freeform':
        z1 = z1.view(1, -1)
        z2 = z2.view(1, -1)
    if model_type == 'stylegan2':
        if type(z1) is list:
            z1 = [z1[0].view(1, -1), z1[1].view(1, -1)]
        else:
            z1 = [z1.view(1, -1), z1.view(1, -1)]
        if type(z2) is list:
            z2 = [z2[0].view(1, -1), z2[1].view(1, -1)]
        else:
            z2 = [z2.view(1, -1), z2.view(1, -1)]

    for n in range(num_interp):
        if model_type == 'freeform':
            interp_zs.append( z2*w.ease(n) + z1*(1-w.ease(n)) )
        if model_type == 'stylegan2':
            interp_zs.append( [ z2[0]*w.ease(n) + z1[0]*(1-w.ease(n)),
                                z2[1]*w.ease(n) + z1[1]*(1-w.ease(n)) ] )
    return interp_zs

@torch.no_grad()
def net_generate(netG, z, model_type='freeform', im_size=512):
    
    if model_type == 'stylegan2':
        z_contents = []
        z_styles = []
        for zidx in range(len(z)):
            z_contents.append(z[zidx][0])
            z_styles.append(z[zidx][1])
        z = [ torch.cat(z_contents), torch.cat(z_styles) ]
        gimg = netG( z, inject_index=8, input_is_latent=True, randomize_noise=False )[0].cpu()
    elif model_type == 'freeform':
        z = torch.cat(z)
        gimg = netG(z)[0].cpu()

    return torch.nn.functional.interpolate(gimg, im_size)

def batch_generate_and_save(netG, zs, folder_name, batch_size=1, model_type='freeform', im_size=1024):
    # zs is a list of vectors if model is freeform
    # zs is a list of lists, each list is 2 vectors, if model is stylegan
    t = 0
    num = 0
    if len(zs) < batch_size:
        gimgs = net_generate(netG, zs, model_type, im_size=im_size).cpu()
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.png"%(num) )
            num += 1

    for k in tqdm(range(len(zs)//batch_size)):
        gimgs = net_generate(netG, zs[k*batch_size:(k+1)*batch_size], model_type, im_size=im_size)
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.png"%(num) )
            num += 1
        t = k

    if len(zs)%batch_size>0:
        gimgs = net_generate(netG, zs[(t+1)*batch_size:], model_type, im_size=im_size)        
        for image in gimgs:
            vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.png"%(num) )
            num += 1



def batch_save(images, folder_name, start_num=0):
    os.makedirs(folder_name, exist_ok=True)
    num = start_num
    for image in images:
        vutils.save_image( image.add(1).mul(0.5), folder_name+"/%d.png"%(num) )
        num += 1


def read_img_and_make_video(dist, video_name, fps):
    img_array = []
    for i in tqdm(range(len(os.listdir(dist)))):
        try:
            filename = dist+'/%d.png'%(i)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        except:
            print('error at: %d'%i)
    
    if '.mp4' not in video_name:
        video_name += '.mp4'
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

from shutil import rmtree

def make_video_from_latents(net, selected_latents, frames_dist_folder, video_name, fps, video_length, ease_fn, model_type, im_size=512):
    # selected_latents: the latent noise of user selected key-frame images, it is a list
    # each item in the list is a vector if the model is freeform, 
    # each item in the list is a list of two vectors if the model is stylegan2
    # frames_dist_folder: the folder path to save the generated images to make the video
    # fps: is the frames we generate per second
    # video_length: is the time of the video, in seconds. For example: 30 means a video length of 30 seconds
    # ease_fn: user selected type of transitions between each key-frame

    # first calculate how many images need to generate
    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    nbr_generate = fps*video_length
    nbr_keyframe = len(selected_latents)
    nbr_interpolation = 1 + nbr_generate // (nbr_keyframe - 1)
    

    main_zs = []
    for idx in range(nbr_keyframe-1):
        main_zs += interpolate_ease_inout(selected_latents[idx], 
            selected_latents[idx+1], nbr_interpolation, ease_fn, model_type)
    

    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=2, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)

@click.command()
@click.pass_context
@click.option('--ckpt', help='Path to checkpoint file')
@click.option('--model_type', help='Model type', type=click.Choice(['stylegan2', 'freeform']), default='freeform')
@click.option('--im_size', help='Image size', default=512)
@click.option('--seeds', help='''Comma separated list of seeds 'a,b,c' ''')
def generate_video(
    ctx: click.Context,
    ckpt: str,
    model_type: str,
    im_size: int,
    seeds: str
):
    if ckpt is None:
        dir = 'C:/Users/natha/repos/FastGAN-pytorch/train_results/NathanGAN/models/'
        ckpt = dir + os.listdir(dir)[0]
    print('checkpoint path is ' + ckpt)
    if not os.path.exists(ckpt):
        ctx.fail('ckpt file at '+ ckpt +' does not exist')

    device = torch.device('cuda')

    from models import Generator as Generator_freeform
    
    frames_dist_folder = 'generated_video_frames' # a folder to save generated images

    video_name = 'generated_video'  # name of the generated video
    noise_dim = 256
    #BEGIN ASSUMPTION: FREEFORM MODEL 
    if model_type == 'freeform':
        net = Generator_freeform(ngf=64, nz=noise_dim, nc=3, im_size=512)
        ### replaced this line with next three from eval.py
        # net.load_state_dict(torch.load(ckpt_path)['g'])
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net.load_state_dict(checkpoint['g'])

        net.to(device)
        net.eval()
    #Only for stylegan2 models which do not work yet
    # else:
    #     with dnnlib.util.open_url(ckpt) as f:
    #         net = legacy.load_network_pkl(f)['G_ema'].to(device)

    try:
        rmtree(frames_dist_folder)
    except:
        pass
    os.makedirs(frames_dist_folder, exist_ok=True)

    fps = 30
    steps = 30
    
    ease_fn=ease_fn_dict['SineEaseInOut']

    ##INSERT SEEDED NOISES
    if seeds is None:
        noises = torch.randn(len(seeds), 256).to(device)
        user_selected_noises = [n for n in noises]
    else:
        seeds = [int(x) for x in seeds.split(',')]
        user_selected_noises = [seed for seed in seeds]
        for i, seed in enumerate(seeds):
            user_selected_noises[i] = torch.tensor(seed2vec(seed, noise_dim), dtype=torch.float32).to(device)
    
    nbl = [steps]*len(seeds)
 
    main_zs = []
    for idx in range(len(user_selected_noises)-1):
        main_zs += interpolate_ease_inout(user_selected_noises[idx], 
                            user_selected_noises[idx+1], nbl[idx], ease_fn, model_type)
    #duplicate the last frame
    main_zs.append(main_zs[-1])
    print('generating images ...')
    batch_generate_and_save(net, main_zs, folder_name=frames_dist_folder, batch_size=1, model_type=model_type, im_size=im_size)
    print('making videos ...')
    read_img_and_make_video(frames_dist_folder, video_name, fps=fps)

if __name__ == "__main__":
    generate_video()