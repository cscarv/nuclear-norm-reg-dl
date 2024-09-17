import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
import torchvision
import samplers
import numpy as np
import importlib
import os
from tqdm import tqdm
from  network_unet import UNet
importlib.reload(samplers)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# L-layer MLP

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.elu(self.layers[i](x))
        return self.layers[-1](x).reshape(-1, 3, 80, 80) # was 128x128
    
# Class Enc1 is just an input layer plus Elu

class Enc1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Enc1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.elu(self.linear(x))

# Class enc2 is just an output layer

class Enc2(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Enc2, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class TruncatedDCT(nn.Module):
    def __init__(self, num_coeffs):
        super(TruncatedDCT, self).__init__()
        self.num_coeffs = num_coeffs

    def forward(self, x):
        x_dct = dct.dct_2d(x)
        x_dct_trunc = x_dct[:,:,:self.num_coeffs,:self.num_coeffs]
        return x_dct_trunc

class TruncatedIDCT(nn.Module):
    def __init__(self, num_coeffs, original_size):
        super(TruncatedIDCT, self).__init__()
        self.num_coeffs = num_coeffs
        self.original_size = original_size

    def forward(self, X_dct_trunc):
        X_dct_trunc = torch.cat([X_dct_trunc, torch.zeros(X_dct_trunc.shape[0], X_dct_trunc.shape[1], self.original_size - self.num_coeffs, self.num_coeffs, device=device)], dim=2)
        X_dct_trunc = torch.cat([X_dct_trunc, torch.zeros(X_dct_trunc.shape[0], X_dct_trunc.shape[1], self.original_size, self.original_size - self.num_coeffs, device=device)], dim=3)
        x_idct = dct.idct_2d(X_dct_trunc)
        return x_idct
    
# Define sparsity-inducing regression loss function

def low_noise_sparsity_inducing_loss(X, X_im, enc1, enc2, dec, sigma):
    fixed_noise_sigma = 1e-3
    B = X.shape[0]

    # Reconstruction loss
    rec_clean = dec(enc2(enc1(X)))
    mse = 0.5 * (1 / B) * torch.sum((rec_clean - X_im)**2)

    # Encoder1 regularization
    image_noise = torch.randn_like(X) * fixed_noise_sigma
    enc1_noisy = enc1(X + image_noise)
    enc1_noiseless = enc1(X)
    reg_enc1 = torch.sum((enc1_noisy - enc1_noiseless)**2)

    # Encoder2 regularization
    latent_clean = enc2(enc1_noiseless)
    enc2_noise = torch.randn_like(enc1_noiseless) * fixed_noise_sigma
    enc2_noisy = enc2(enc1_noiseless + enc2_noise)
    reg_enc2 = torch.sum((enc2_noisy - latent_clean)**2)

    # Scale regularization loss
    reg = (sigma**2)/(fixed_noise_sigma**2) * (1 / B) * 0.5 * (reg_enc1 + reg_enc2)

    return mse + reg

def main():
    # Load CelebA
    to_tensor = torchvision.transforms.ToTensor()
    downsize = torchvision.transforms.Resize((256, 256)) # was 512x512
    composed_transform = torchvision.transforms.Compose([downsize, to_tensor])
    root = "" # Fill in the path to the CelebA dataset
    trainset = torchvision.datasets.CelebA(root=root, split='train', download=True, transform=composed_transform)
    batch_size = 16
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Train encoder-decoder model using sparsity-inducing loss

    D = 3*80*80
    input_dim = D
    latent_dim = 700
    hidden_dim = 10000

    num_coeffs = 80

    trunc_dct = TruncatedDCT(num_coeffs=num_coeffs)
    trunc_idct = TruncatedIDCT(num_coeffs=num_coeffs, original_size=256)

    enc1 = Enc1(input_dim, hidden_dim).to(device)
    enc2 = Enc2(hidden_dim, latent_dim).to(device)
    output_unet = UNet(in_nc=3, out_nc=3).to(device)
    dec = nn.Sequential(MLP(latent_dim,input_dim,[hidden_dim]), trunc_idct, output_unet).to(device)

    sigma = 0.5
    lr = 1e-4
    start_epoch = 0
    num_epochs = 4

    optimizer = torch.optim.AdamW(list(enc1.parameters()) + list(enc2.parameters()) + list(dec.parameters()), lr=lr, weight_decay=0)

    losses = []

    # Directory to save checkpoints
    checkpoint_dir = '/checkpoints/celeba_autoencoder_sigma_0p5'
    # Make the directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print('Epoch:', epoch)
        for i, (X, _) in enumerate(trainloader):
            X = X.to(device)
            # Take truncated DCT of X
            X_dct = trunc_dct(X) 
            X_dct = X_dct.reshape(X_dct.shape[0], D)
            optimizer.zero_grad()
            loss = low_noise_sparsity_inducing_loss(X_dct, X, enc1, enc2, dec, sigma)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0:
                print('Epoch: %d, Iter: %d, Loss: %.4f' % (epoch, i, loss.item()))
        
        # Save checkpoint
        if epoch % 1 == 0:
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'enc1_state_dict': enc1.state_dict(),
                'enc2_state_dict': enc2.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                # }, checkpoint_dir + '/checkpoint_epoch' + str(epoch) + '.pt')
                }, checkpoint_dir + '/checkpoint_last_epoch.pt')
        
if __name__ == '__main__':
    main()