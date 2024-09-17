import torch
import torch.nn as nn
import torch.nn.functional as F
import basicblock as B
from  network_unet import UNetEncoder, UNetDecoder
import importlib
import os
from tqdm import tqdm

import imagenet.imagenet as imagenet
importlib.reload(imagenet)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define Noise2Noise loss function

def noise2noise_loss(clean_ims, enc, dec, sigma):
    # clean_ims is n x d
    # enc is encoder MLP
    # dec is decoder MLP

    batch_size = clean_ims.shape[0]

    # Generate noise
    source_noise = torch.randn_like(clean_ims) * sigma
    target_noise = torch.randn_like(clean_ims) * sigma

    # Add noise to source and targets
    noisy_targets = clean_ims + target_noise
    noisy_sources = clean_ims + source_noise

    # Compute reconstruction loss
    rec = dec(enc(noisy_sources))
    mse = 0.5 * (1 / batch_size) * torch.sum((rec - noisy_targets)**2)

    return mse

def main():
    # Load dataset
    dataset = imagenet.ImageNetSRTrain(size=128, degradation='pil_bicubic')
    # Construct subset of dataset
    n_ims = len(dataset)
    dataset = torch.utils.data.Subset(dataset, range(int(n_ims/4)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1) # was 32

    lr = 1e-4

    enc_model_noise2noise = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_noise2noise = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_noise2noise.train()
    dec_model_noise2noise.train()
    optimizer = torch.optim.AdamW(list(enc_model_noise2noise.parameters()) + list(dec_model_noise2noise.parameters()), lr=lr)

    # Load checkpoint
    # fname = "" # Fill in the path to the checkpoint if you want to resume training
    # checkpoint = torch.load(fname, map_location=device)
    # enc_model_noise2noise.load_state_dict(checkpoint['enc_model_noise2noise_state_dict'])
    # dec_model_noise2noise.load_state_dict(checkpoint['dec_model_noise2noise_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # noise2noise_losses = checkpoint['noise2noise_losses']

    noise2noise_losses = []
    start_epoch = 0
    n_epochs = 2

    # Directory to save checkpoints
    checkpoint_dir = '/checkpoints/imagenet_sigma2_noise2noise'
    # Make the directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        print('Epoch {}'.format(epoch))
        for i, data in tqdm(enumerate(dataloader)):
            # get the inputs
            inputs = data['image'].permute(0, 3, 1, 2).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            sigma = 2.0
            loss = noise2noise_loss(inputs, enc_model_noise2noise, dec_model_noise2noise, sigma)
            loss.backward()
            optimizer.step()

            # print statistics
            noise2noise_losses.append(loss.item())
            if i % 10 == 0:
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, noise2noise_losses[-1]))
        checkpoint_fname = os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pt'.format(epoch))
        torch.save({
            'enc_model_noise2noise_state_dict': enc_model_noise2noise.state_dict(),
            'dec_model_noise2noise_state_dict': dec_model_noise2noise.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'noise2noise_losses': noise2noise_losses
        }, checkpoint_fname)

    print('Finished Training')

if __name__ == '__main__':
    main()