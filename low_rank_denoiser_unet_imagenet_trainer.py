import torch
from  network_unet import UNetEncoder, UNetDecoder
import importlib
import os
from tqdm import tqdm

# import imagenet.imagenet as imagenet
# importlib.reload(imagenet)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def low_noise_sparsity_inducing_loss(X, enc, dec, sigma):
    fixed_noise_sigma = 1e-3
    B = X.shape[0]

    # Reconstruction loss
    rec_clean = dec(enc(X))
    mse = 0.5 * (1 / B) * torch.sum((rec_clean - X)**2)

    # Encoder regularization
    image_noise = torch.randn_like(X) * fixed_noise_sigma
    enc_noisy = enc(X + image_noise)
    enc_noiseless = enc(X)
    reg_enc = torch.sum((enc_noisy - enc_noiseless)**2)

    # Decoder regularization
    latent = enc(X)
    latent_noise = torch.randn_like(latent) * fixed_noise_sigma
    rec_noisy = dec(latent + latent_noise)
    reg_dec = torch.sum((rec_noisy - rec_clean)**2)

    # Scale regularization loss
    reg = (sigma**2)/(fixed_noise_sigma**2) * 0.5 * (1 / B) * (reg_enc + reg_dec)

    return mse + reg

def main():
    # Load dataset
    # dataset = imagenet.ImageNetSRTrain(size=128, degradation='pil_bicubic')
    dataset = None ### LOAD YOUR DATASET HERE ###
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1) # was 32

    lr = 1e-4

    enc_model_sparse = UNetEncoder(in_nc=3, out_nc=3).to(device)
    dec_model_sparse = UNetDecoder(in_nc=3, out_nc=3).to(device)
    enc_model_sparse.train()
    dec_model_sparse.train()
    optimizer = torch.optim.AdamW(list(enc_model_sparse.parameters()) + list(dec_model_sparse.parameters()), lr=lr)

    # Load checkpoint
    # fname = "" # Fill in the path to the checkpoint if you want to resume training
    # checkpoint = torch.load(fname, map_location=device)
    # enc_model_sparse.load_state_dict(checkpoint['enc_model_sparse_state_dict'])
    # dec_model_sparse.load_state_dict(checkpoint['dec_model_sparse_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # sparse_losses = checkpoint['sparse_losses']

    sparse_losses = []
    start_epoch = 0
    n_epochs = 2

    # Directory to save checkpoints
    checkpoint_dir = '/checkpoints/imagenet_sigma2_our_denoiser'
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
            # Add noise to inputs
            inputs_noisy = inputs + torch.randn_like(inputs) * sigma
            loss = low_noise_sparsity_inducing_loss(inputs_noisy, enc_model_sparse, dec_model_sparse, sigma)
            loss.backward()
            optimizer.step()

            # print statistics
            sparse_losses.append(loss.item())
            if i % 10 == 0:
                print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, sparse_losses[-1]))
        checkpoint_fname = os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pt'.format(epoch))
        torch.save({
            'enc_model_sparse_state_dict': enc_model_sparse.state_dict(),
            'dec_model_sparse_state_dict': dec_model_sparse.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sparse_losses': sparse_losses
        }, checkpoint_fname)

    print('Finished Training')

if __name__ == '__main__':
    main()