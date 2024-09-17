import torch
import torchvision
import torch.nn as nn
from  network_unet import UNetEncoder, UNetDecoder
import numpy as np
import os
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
import bm3d
import imagenet.imagenet as imagenet

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def compute_psnr(model, image, noise_std):
    psnr_fn = PSNR().to(device)
    image = image.to(device)
    noise = torch.randn_like(image) * noise_std
    noisy_image = image + noise
    denoised_image = model(noisy_image)
    psnr = psnr_fn(denoised_image, image).cpu().numpy()
    return psnr, noisy_image.detach(), denoised_image.detach()

def compute_psnr_bm3d(image, noise_std):
    psnr_fn = PSNR().to(device)
    image = image.to(device)
    noise = torch.randn_like(image) * noise_std
    noisy_image = image + noise
    noisy_image = noisy_image.permute(1, 2, 0).cpu().numpy()
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=noise_std, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    denoised_image = torch.tensor(denoised_image).permute(2, 0, 1).to(device)
    noisy_image = torch.tensor(noisy_image).permute(2, 0, 1).to(device)
    psnr = psnr_fn(denoised_image, image).cpu().numpy()
    return psnr, noisy_image, denoised_image

def normalize_image(im):
    return (im + 1) / 2.0

def experiments(data, sigma):
       
    # Define directories for saving results

    save_dir_imagenet = '/results/imagenet-256x256'
    save_dir_imagenet_sigma = os.path.join(save_dir_imagenet, 'sigma{}'.format(sigma))
    save_dir_imagenet_ours = os.path.join(save_dir_imagenet_sigma, 'ours')
    save_dir_imagenet_ordinary = os.path.join(save_dir_imagenet_sigma, 'ordinary')
    save_dir_imagenet_noise2noise = os.path.join(save_dir_imagenet_sigma, 'noise2noise')
    save_dir_imagenet_bm3d = os.path.join(save_dir_imagenet_sigma, 'bm3d')

    os.makedirs(save_dir_imagenet_sigma, exist_ok=True)
    os.makedirs(save_dir_imagenet_ours, exist_ok=True)
    os.makedirs(save_dir_imagenet_ordinary, exist_ok=True)
    os.makedirs(save_dir_imagenet_noise2noise, exist_ok=True)
    os.makedirs(save_dir_imagenet_bm3d, exist_ok=True)

    # Load our model -- name based on sigma
    fname = "" # Fill in the path to the checkpoint
    checkpoint = torch.load(fname, map_location=device)

    enc_model_sparse = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_sparse = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_sparse.eval()
    dec_model_sparse.eval()

    enc_model_sparse.load_state_dict(checkpoint['enc_model_sparse_state_dict'])
    dec_model_sparse.load_state_dict(checkpoint['dec_model_sparse_state_dict'])

    model = nn.Sequential(enc_model_sparse, dec_model_sparse).to(device)

    # Compute PSNR for each image
    psnrs_ours = []
    for i, img in tqdm(enumerate(data)):
        with torch.no_grad():
            psnr, noisy_image, denoised_image = compute_psnr(model, img, sigma)
            psnrs_ours.append(psnr)
            # Save clean, noisy and denoised images
            torchvision.utils.save_image(normalize_image(img), os.path.join(save_dir_imagenet_ours, 'clean_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(noisy_image), os.path.join(save_dir_imagenet_ours, 'noisy_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(denoised_image), os.path.join(save_dir_imagenet_ours, 'denoised_image_{}.png'.format(i)))
            del noisy_image, denoised_image
    # Display average PSNR
    print('Average PSNR for our model: {}'.format(np.mean(psnrs_ours)))
    # Save PSNRs
    np.save(os.path.join(save_dir_imagenet_ours, 'psnrs.npy'), psnrs_ours)

    # Delete models and checkpoint

    del model, enc_model_sparse, dec_model_sparse, checkpoint

    # Load ordinary model
    fname = "" # Fill in the path to the checkpoint
    checkpoint = torch.load(fname, map_location=device)

    enc_model_ordinary = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_ordinary = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_ordinary.eval()
    dec_model_ordinary.eval()

    enc_model_ordinary.load_state_dict(checkpoint['enc_model_ordinary_state_dict'])
    dec_model_ordinary.load_state_dict(checkpoint['dec_model_ordinary_state_dict'])

    model = nn.Sequential(enc_model_ordinary, dec_model_ordinary).to(device)

    # Compute PSNR for each image
    psnrs_ordinary = []
    for i, img in tqdm(enumerate(data)):
        with torch.no_grad():
            psnr, noisy_image, denoised_image = compute_psnr(model, img, sigma)
            psnrs_ordinary.append(psnr)
            # Save clean, noisy and denoised images
            torchvision.utils.save_image(normalize_image(img), os.path.join(save_dir_imagenet_ordinary, 'clean_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(noisy_image), os.path.join(save_dir_imagenet_ordinary, 'noisy_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(denoised_image), os.path.join(save_dir_imagenet_ordinary, 'denoised_image_{}.png'.format(i)))
            del noisy_image, denoised_image
    # Display average PSNR
    print('Average PSNR for ordinary model: {}'.format(np.mean(psnrs_ordinary)))
    # Save PSNRs
    np.save(os.path.join(save_dir_imagenet_ordinary, 'psnrs.npy'), psnrs_ordinary)

    # Delete models and checkpoint

    del model, enc_model_ordinary, dec_model_ordinary, checkpoint

    # Load noise2noise model
    fname = "" # Fill in the path to the checkpoint
    checkpoint = torch.load(fname, map_location=device)

    enc_model_noise2noise = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_noise2noise = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_noise2noise.eval()
    dec_model_noise2noise.eval()

    enc_model_noise2noise.load_state_dict(checkpoint['enc_model_noise2noise_state_dict'])
    dec_model_noise2noise.load_state_dict(checkpoint['dec_model_noise2noise_state_dict'])

    model = nn.Sequential(enc_model_noise2noise, dec_model_noise2noise).to(device)

    # Compute PSNR for each image
    psnrs_noise2noise = []
    for i, img in tqdm(enumerate(data)):
        with torch.no_grad():
            psnr, noisy_image, denoised_image = compute_psnr(model, img, sigma)
            psnrs_noise2noise.append(psnr)
            # Save clean, noisy and denoised images
            torchvision.utils.save_image(normalize_image(img), os.path.join(save_dir_imagenet_noise2noise, 'clean_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(noisy_image), os.path.join(save_dir_imagenet_noise2noise, 'noisy_image_{}.png'.format(i)))
            torchvision.utils.save_image(normalize_image(denoised_image), os.path.join(save_dir_imagenet_noise2noise, 'denoised_image_{}.png'.format(i)))
            del noisy_image, denoised_image
    # Display average PSNR
    print('Average PSNR for noise2noise model: {}'.format(np.mean(psnrs_noise2noise)))
    # Save PSNRs
    np.save(os.path.join(save_dir_imagenet_noise2noise, 'psnrs.npy'), psnrs_noise2noise)

    # Delete models and checkpoint

    del model, enc_model_noise2noise, dec_model_noise2noise, checkpoint

    # Compute PSNR for BM3D
    psnrs_bm3d = []
    for i, img in tqdm(enumerate(data)):
        psnr, noisy_image, denoised_image = compute_psnr_bm3d(img, sigma)
        print("PSNR of BM3D on image {}: {}".format(i, psnr))
        psnrs_bm3d.append(psnr)
        # Save clean, noisy and denoised images
        torchvision.utils.save_image(normalize_image(img), os.path.join(save_dir_imagenet_bm3d, 'clean_image_{}.png'.format(i)))
        torchvision.utils.save_image(normalize_image(noisy_image), os.path.join(save_dir_imagenet_bm3d, 'noisy_image_{}.png'.format(i)))
        torchvision.utils.save_image(normalize_image(denoised_image), os.path.join(save_dir_imagenet_bm3d, 'denoised_image_{}.png'.format(i)))
        del noisy_image, denoised_image
    # Display average PSNR
    print('Average PSNR for BM3D: {}'.format(np.mean(psnrs_bm3d)))
    # Save PSNRs
    np.save(os.path.join(save_dir_imagenet_bm3d, 'psnrs.npy'), psnrs_bm3d)

def main():
    # Load data -- 100 random images from ImageNet validation set
    dataset = imagenet.ImageNetSRValidation(size=256, degradation='pil_bicubic')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
    data = next(iter(dataloader))['image'].permute(0, 3, 1, 2)

    # Run experiments for each sigma

    sigmas = [1, 2]
    for sigma in sigmas:
        experiments(data, sigma)

if __name__ == '__main__':
    main()