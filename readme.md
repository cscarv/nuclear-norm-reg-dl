This is code to accompany "Nuclear Norm Regularization for Deep Learning."

We have included a requirements.txt file to document the required packages.

### Validation experiments (Section 4)

- "rof_ours_vs_exact_experiments.py" solves problems (8) and (9) for the 2-dimensional problems.
- "rof_ours_vs_exact_experiments_d5.py" solves problems (8) and (9) for the 5-dimensional problems.
- "rof_ours_vs_exact_plot_generator.ipynb" includes code to help generate the plots from Section 4.

### Denoising experiments (Section 5)

- "low_rank_denoiser_unet_imagenet_trainer.py" trains our proposed denoiser.
- "ordinary_denoiser_unet_imagenet_trainer.py" trains the supervised denoiser.
- "noise2noise_denoiser_unet_imagenet_trainer.py" trains the Noise2Noise baseline (Lehtinen et al., 2018).
- "denoising_benchmarks_imagenet.py" runs PSNR benchmarks on 100 randomly-chosen images from the Imagenet validation set.
- "denoising_benchmarks_CBSD68.py" runs PSNR benchmarks on the popular CBSD68 test set. These test images are included in the directory "CBSD68".
- "basicblock.py" and "network_unet.py" are lightly modified versions of code from the [Deep Plug-and-Play Image Restoration Github repo](https://github.com/cszn/DPIR.git).

### Representation learning experiments (Section 5)

- "final-celeba-autoencoder.py" trains our proposed autoencoder -- set sigma=0.5 in the code to train our regularized autoencoder, and sigma=0 to train the unregularized autoencoder.
- "celeba-autoencoder-visualizations.ipynb" includes code to help generate the latent traversals for our autoencoders in Figures 6 and 7.
- "celeba-beta-vae-visualizations.ipynb" includes code to help generate the latent traversals for the beta-VAE in Figure 8.

Thank you for your interest in our submission!