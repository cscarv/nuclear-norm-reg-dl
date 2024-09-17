import torch
import torch.nn as nn
import torch.nn.functional as F
import basicblock as B
from  network_unet import UNetEncoder, UNetDecoder
import numpy as np
import importlib
import os
from tqdm import tqdm

import imagenet.imagenet as imagenet
importlib.reload(imagenet)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define ordinary denoising loss

def ordinary_denoising_loss(X, enc, dec, sigma):
    # X is n x d
    # enc is encoder MLP
    # dec is decoder MLP

    batch_size = X.shape[0]

    # Generate noise
    Z = torch.randn_like(X) * sigma

    # Compute reconstruction loss
    rec = dec(enc(X + Z))
    mse = 0.5 * (1 / batch_size) * torch.sum((rec - X)**2)

    return mse

def main():
    # Load dataset
    dataset = imagenet.ImageNetSRTrain(size=128, degradation='pil_bicubic')
    # Construct subset of dataset
    n_ims = len(dataset)
    dataset = torch.utils.data.Subset(dataset, range(int(n_ims/4)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1) # was 32

    lr = 1e-4

    enc_model_ordinary = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_ordinary = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_ordinary.train()
    dec_model_ordinary.train()
    optimizer = torch.optim.AdamW(list(enc_model_ordinary.parameters()) + list(dec_model_ordinary.parameters()), lr=lr)

    # Load checkpoint
    # fname = "" # Fill in the path to the checkpoint if you want to resume training
    # checkpoint = torch.load(fname, map_location=device)
    # enc_model_ordinary.load_state_dict(checkpoint['enc_model_ordinary_state_dict'])
    # dec_model_ordinary.load_state_dict(checkpoint['dec_model_ordinary_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # ordinary_losses = checkpoint['ordinary_losses']

    ordinary_losses = []
    start_epoch = 0
    n_epochs = 2

    # Directory to save checkpoints
    checkpoint_dir = '/checkpoints/imagenet_sigma2_supervised_denoiser'
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
            loss = ordinary_denoising_loss(inputs, enc_model_ordinary, dec_model_ordinary, sigma=sigma)
            loss.backward()
            optimizer.step()

            # print statistics
            ordinary_losses.append(loss.item())
            if i % 10 == 0:
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, ordinary_losses[-1]))
        checkpoint_fname = os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pt'.format(epoch))
        torch.save({
            'enc_model_ordinary_state_dict': enc_model_ordinary.state_dict(),
            'dec_model_ordinary_state_dict': dec_model_ordinary.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ordinary_losses': ordinary_losses
        }, checkpoint_fname)

    print('Finished Training')

if __name__ == '__main__':
    main()