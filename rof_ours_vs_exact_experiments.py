import torch
import torch.nn as nn
import torch.nn.functional as F
import samplers
import numpy as np
import importlib
import os
from functorch import vmap, jacrev
from tqdm import tqdm
importlib.reload(samplers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InputMapping(nn.Module):
    """Fourier features mapping."""

    def __init__(
        self, d_in, n_freq, sigma=1, tdiv=2, incrementalMask=True, Tperiod=None, kill=False
    ):
        super().__init__()
        Bmat = torch.randn(n_freq, d_in) * np.pi * sigma / np.sqrt(d_in)  # gaussian
        # time frequencies are a quarter of spacial frequencies.
        # Bmat[:, d_in-1] /= tdiv
        Bmat[:, 0] /= tdiv

        self.Tperiod = Tperiod
        if Tperiod is not None:
            # Tcycles = (Bmat[:, d_in-1]*Tperiod/(2*np.pi)).round()
            # K = Tcycles*(2*np.pi)/Tperiod
            # Bmat[:, d_in-1] = K
            Tcycles = (Bmat[:, 0] * Tperiod / (2 * np.pi)).round()
            K = Tcycles * (2 * np.pi) / Tperiod
            Bmat[:, 0] = K

        Bnorms = torch.norm(Bmat, p=2, dim=1)
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices, :]

        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in if Tperiod is None else n_freq * 2 + d_in - 1
        self.B = nn.Linear(d_in, self.d_out, bias=False)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
            self.mask = nn.Parameter(torch.zeros(1, n_freq), requires_grad=False)

        self.incrementalMask = incrementalMask
        if not incrementalMask:
            self.mask = nn.Parameter(torch.ones(1, n_freq), requires_grad=False)
        if kill:
            self.mask = nn.Parameter(torch.zeros(1, n_freq), requires_grad=False)

    def step(self, progressPercent):
        if self.incrementalMask:
            float_filled = (progressPercent * self.n_freq) / 0.7
            int_filled = int(float_filled // 1)
            # remainder = float_filled % 1

            if int_filled >= self.n_freq:
                self.mask[0, :] = 1
            else:
                self.mask[0, 0:int_filled] = 1
                # self.mask[0, int_filled] = remainder

    def forward(self, xi):
        # pdb.set_trace()
        dim = self.d_in - 1 # was xi.shape[1] - 1
        y = self.B(xi)
        # Unsqueeze y and xi at dim=0 if they are 1D tensors
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        if len(xi.shape) == 1:
            xi = xi.unsqueeze(0)
        if self.Tperiod is None:
            return torch.cat([torch.sin(y) * self.mask, torch.cos(y) * self.mask, xi], dim=-1)
        else:
            return torch.cat(
                [torch.sin(y) * self.mask, torch.cos(y) * self.mask, xi[:, 1 : dim + 1]], dim=-1
            )
    
# Define a two-layer MLP with output_dim=hidden_dims -- this is our "h" function

class MLPh(nn.Module):
    def __init__(self, base_dims, hidden_dims, fourier_map=None):
        super(MLPh, self).__init__()
        self.fourier_map = fourier_map
        self.base_dims = base_dims
        if self.fourier_map is not None:
            self.base_dims = fourier_map.d_out
            print(self.base_dims)
        self.fc1 = nn.Linear(self.base_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x):
        if self.fourier_map is not None:
            x = self.fourier_map(x)
        x = F.elu(self.fc1(x)) # elu works well!
        x = self.fc2(x)
        return x
    
# Define a two-layer MLP with output_dim=1 -- this is our "g" function

class MLPg(nn.Module):
    def __init__(self, hidden_dims, out_dims=1):
        super(MLPg, self).__init__()
        self.fc1 = nn.Linear(hidden_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        x = F.elu(self.fc1(x)) # elu works well!
        x = self.fc2(x)
        return x.squeeze()
    
# def denoising_regularizer(model, x, eta, eta_base=1e-3):
#     # Compute model(x + eta_base*noise) - model(x)
#     noise = torch.randn_like(x) * eta_base
#     noisy_outs = model(x + noise)
#     clean_outs = model(x)
#     n = x.shape[0]
#     return (eta)/(eta_base ** 2) * (1 / n) * torch.sum((noisy_outs - clean_outs) ** 2)

def denoising_regularizer(model, x, eta, eta_base=1e-3):
    # Compute model(x + eta_base*noise) - model(x)
    num_noise = 10
    n = x.shape[0]
    noise = torch.randn(num_noise*n, x.shape[-1], device=device) * eta_base
    noisy_outs = model(x.repeat(num_noise , 1) + noise)
    clean_outs = model(x.repeat(num_noise , 1))
    return (eta)/(eta_base ** 2) * (1 / (num_noise * n)) * torch.sum((noisy_outs - clean_outs) ** 2)
    
# Define loss function: ||f(x) - g(x)||^2 + nuclear norm

def our_nuc_reg_loss_function(h_func, g_func, target_fn, x, reg_parameter):
    model = lambda x: g_func(h_func(x))
    target_vals = target_fn(x)
    y = model(x)
    reconstruction_loss = 0.5 * F.mse_loss(y, target_vals)

    # Compute the Jacobian of h_func with respect to the inputs
    h_func.zero_grad()
    g_func.zero_grad()
    x.requires_grad = True
    compute_batch_jacobian_hfunc = vmap(jacrev(h_func, argnums=0), in_dims=(0))
    J_h = compute_batch_jacobian_hfunc(x).squeeze()
    hx = h_func(x)
    compute_batch_jacobian_gfunc = vmap(jacrev(g_func, argnums=0), in_dims=(0))
    J_g = compute_batch_jacobian_gfunc(hx)
    # Compute sq Frobenius norm of both Jacobians
    Jh_norm = torch.sum(J_h ** 2, dim=(1, 2))
    Jh_norm = Jh_norm.mean()
    Jg_norm = torch.sum(J_g ** 2, dim=(1))
    Jg_norm = Jg_norm.mean()
    our_nuc_norm = 0.5 * (Jh_norm + Jg_norm)

    total_loss = reconstruction_loss + reg_parameter * our_nuc_norm

    return total_loss, our_nuc_norm

def our_nuc_reg_denoising_loss_function(h_func, g_func, target_fn, x, reg_parameter):
    model = lambda x: g_func(h_func(x))
    target_vals = target_fn(x)
    y = model(x)
    reconstruction_loss = 0.5 * F.mse_loss(y, target_vals)

    # Compute denoising regularizer of h_func at x and g_func at h_func(x)
    denoising_reg_h = denoising_regularizer(h_func, x, eta=reg_parameter)
    hx = h_func(x)
    denoising_reg_g = denoising_regularizer(g_func, hx, eta=reg_parameter)
    our_nuc_norm = 0.5 * (denoising_reg_h + denoising_reg_g)

    total_loss = reconstruction_loss + our_nuc_norm

    return total_loss, our_nuc_norm

# Define loss function: ||f(x) - g(x)||^2 + nuclear norm

def exact_nuc_reg_loss_function(model, target_fn, x, reg_parameter):
    target_vals = target_fn(x)
    y = model(x)
    reconstruction_loss = 0.5 * F.mse_loss(y, target_vals)

    # Compute Jacobian of the model wrt the inputs at x
    model.zero_grad()
    x.requires_grad = True
    compute_batch_jacobian = vmap(jacrev(model, argnums=0), in_dims=(0))
    J = compute_batch_jacobian(x) # shape (batch_size, out_dims, in_dims)
    grads = J
    # Compute the norm of the gradient
    grad_norm = torch.linalg.vector_norm(grads, ord=2, dim=1) # shape (batch_size,)
    nuc_norm = grad_norm.mean()
    
    total_loss = reconstruction_loss + reg_parameter * nuc_norm

    return total_loss, nuc_norm

# Target function is indicator of ball of radius R in 2D

def target_fn(x, radius=1):
    return (torch.linalg.norm(x, dim=1) <= radius).float()
    # Indicator of unit square in 2D
    # return (torch.abs(x[:, 0]) <= radius).float() * (torch.abs(x[:, 1]) <= radius).float()
    # Indicator of union of two unit squares in 2D
    # return (torch.abs(x[:, 0] - 0.5) <= radius).float() * (torch.abs(x[:, 1] - 0.5) <= radius).float() + (torch.abs(x[:, 0] + 0.5) <= radius).float() * (torch.abs(x[:, 1] + 0.5) <= radius).float()

# Trainer for exact objective

def train_exact_objective(target_fn, fourier_map, reg_parameter):
    d = 2
    base_dims = d
    hidden_dims = 100
    model_exact_nuc = nn.Sequential(MLPh(base_dims, hidden_dims, fourier_map), MLPg(hidden_dims)).to(device)
    optimizer_exact = torch.optim.AdamW(model_exact_nuc.parameters(), lr=1e-4)

    exact_nuc_losses = []
    exact_nuc_vals = []
    avg_values_exact_nuc = []
    outside_values_exact_nuc = []
    abs_errors_exact_nuc = []

    n_iter = 100000 # push this very far
    batch_size = 100000 # 100k now

    print("Training exact objective with reg parameter " + str(reg_parameter))

    for i in tqdm(range(n_iter)):
        optimizer_exact.zero_grad()
        X = 20 * torch.rand(batch_size, base_dims, device=device) - 10
        exact_nuc_loss, exact_nuc_reg = exact_nuc_reg_loss_function(model_exact_nuc, target_fn, X, reg_parameter)
        exact_nuc_loss.backward()
        exact_nuc_losses.append(exact_nuc_loss.item())
        exact_nuc_vals.append(exact_nuc_reg.item())
        optimizer_exact.step()
        if i % 100 == 0:
            print(i, exact_nuc_loss.item())
            # Compute average value of function on unit disc
            n_samples = batch_size
            X_disc = 20 * torch.rand(batch_size, base_dims, device=device) - 10
            X_disc = X_disc[torch.linalg.norm(X_disc, dim=1) <= 1]
            y_disc_exact = model_exact_nuc(X_disc).squeeze().detach().cpu()
            avg_val_exact = y_disc_exact.mean()
            avg_values_exact_nuc.append(avg_val_exact.item())
            print("avg val: " + str(avg_val_exact.item()))
            # Compute absolute error between learned and true sol function on unit square
            X_square = 20 * torch.rand(batch_size, base_dims, device=device) - 10
            y_square_exact = model_exact_nuc(X_square).squeeze().detach().cpu()
            y_square_target = (1 - d*reg_parameter) * target_fn(X_square).squeeze().detach().cpu()
            abs_err = torch.mean(torch.abs(y_square_exact - y_square_target))
            abs_errors_exact_nuc.append(abs_err.item())
            print("abs error: " + str(abs_err.item()))
            # Compute average value of function outside the unit disc
            X_outside = 20 * torch.rand(batch_size, base_dims, device=device) - 10
            X_outside = X_outside[torch.linalg.norm(X_outside, dim=1) > 1]
            y_outside_exact = model_exact_nuc(X_outside).squeeze().detach().cpu()
            avg_val_outside_exact = y_outside_exact.mean()
            outside_values_exact_nuc.append(avg_val_outside_exact.item())
            print("avg val outside: " + str(avg_val_outside_exact.item()))

    return model_exact_nuc, exact_nuc_losses, exact_nuc_vals, avg_values_exact_nuc, abs_errors_exact_nuc

# Trainer for our objective

def train_our_objective(target_fn, fourier_map, reg_parameter):
    d = 2
    base_dims = d
    hidden_dims = 100
    # Model is composition of two MLPs
    g_model_our_nuc = MLPg(hidden_dims).to(device)
    h_model_our_nuc = MLPh(base_dims, hidden_dims, fourier_map=fourier_map).to(device)
    optimizer_ours = torch.optim.AdamW(list(h_model_our_nuc.parameters()) + list(g_model_our_nuc.parameters()), lr=1e-4)

    our_nuc_losses = []
    our_nuc_vals = []
    avg_values_our_nuc = []
    outside_values_our_nuc = []
    abs_errors_our_nuc = []

    if reg_parameter <= 0.05: # warm up from 0.05
        n_iter = 100000 # push this very far
        batch_size = 10000

        print("Training our objective with DENOISING and reg parameter " + str(reg_parameter))

        for i in tqdm(range(n_iter)):
            optimizer_ours.zero_grad()
            # Draw samples from [-2, 2]^2
            X = 20 * torch.rand(batch_size, base_dims, device=device) - 10
            our_nuc_loss, our_nuc_reg = our_nuc_reg_denoising_loss_function(h_model_our_nuc, g_model_our_nuc, target_fn, X, reg_parameter)
            our_nuc_loss.backward()
            our_nuc_losses.append(our_nuc_loss.item())
            our_nuc_vals.append(our_nuc_reg.item())
            optimizer_ours.step()
            if i % 100 == 0:
                print(i, our_nuc_loss.item())
                # Compute average value of function on unit disc
                n_samples = batch_size
                X_disc = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                X_disc = X_disc[torch.linalg.norm(X_disc, dim=1) <= 1]
                y_disc_ours = g_model_our_nuc(h_model_our_nuc(X_disc)).squeeze().detach().cpu()
                avg_val_ours = y_disc_ours.mean()
                avg_values_our_nuc.append(avg_val_ours.item())
                print("avg val: " + str(avg_val_ours.item()))
                # Compute absolute error between ours and true sol function on unit square
                X_square = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                y_square_ours = g_model_our_nuc(h_model_our_nuc(X_square)).squeeze().detach().cpu()
                y_square_target = (1 - d*reg_parameter) * target_fn(X_square).squeeze().detach().cpu()
                abs_err = torch.mean(torch.abs(y_square_ours - y_square_target))
                abs_errors_our_nuc.append(abs_err.item())
                print("abs error: " + str(abs_err.item()))
                # Compute average value of function outside the unit disc
                X_outside = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                X_outside = X_outside[torch.linalg.norm(X_outside, dim=1) > 1]
                y_outside_ours = g_model_our_nuc(h_model_our_nuc(X_outside)).squeeze().detach().cpu()
                avg_val_outside_ours = y_outside_ours.mean()
                outside_values_our_nuc.append(avg_val_outside_ours.item())
                print("avg val outside: " + str(avg_val_outside_ours.item()))
        
    else:
        # Warm up with reg_parameter = 0.05 for 10k iterations
        # Then increase reg_parameter by 0.05 every 10k iterations
        total_iter_counter = 0
        n_iter = 10000
        total_iter_counter += n_iter
        batch_size = 10000
        initial_reg_param = 0.05
        print("Training our objective with DENOISING and reg parameter " + str(initial_reg_param))

        for i in tqdm(range(n_iter)):
            optimizer_ours.zero_grad()
            # Draw samples from [-2, 2]^2
            X = 20 * torch.rand(batch_size, base_dims, device=device) - 10
            our_nuc_loss, our_nuc_reg = our_nuc_reg_denoising_loss_function(h_model_our_nuc, g_model_our_nuc, target_fn, X, initial_reg_param)
            our_nuc_loss.backward()
            our_nuc_losses.append(our_nuc_loss.item())
            our_nuc_vals.append(our_nuc_reg.item())
            optimizer_ours.step()
            if i % 100 == 0:
                print(i, our_nuc_loss.item())
                # Compute average value of function on unit disc
                n_samples = batch_size
                X_disc = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                X_disc = X_disc[torch.linalg.norm(X_disc, dim=1) <= 1]
                y_disc_ours = g_model_our_nuc(h_model_our_nuc(X_disc)).squeeze().detach().cpu()
                avg_val_ours = y_disc_ours.mean()
                avg_values_our_nuc.append(avg_val_ours.item())
                print("avg val: " + str(avg_val_ours.item()))
                # Compute absolute error between ours and true sol function on unit square
                X_square = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                y_square_ours = g_model_our_nuc(h_model_our_nuc(X_square)).squeeze().detach().cpu()
                y_square_target = (1 - d*reg_parameter) * target_fn(X_square).squeeze().detach().cpu()
                abs_err = torch.mean(torch.abs(y_square_ours - y_square_target))
                abs_errors_our_nuc.append(abs_err.item())
                print("abs error: " + str(abs_err.item()))
                # Compute average value of function outside the unit disc
                X_outside = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                X_outside = X_outside[torch.linalg.norm(X_outside, dim=1) > 1]
                y_outside_ours = g_model_our_nuc(h_model_our_nuc(X_outside)).squeeze().detach().cpu()
                avg_val_outside_ours = y_outside_ours.mean()
                outside_values_our_nuc.append(avg_val_outside_ours.item())
                print("avg val outside: " + str(avg_val_outside_ours.item()))

        while initial_reg_param < reg_parameter:
            initial_reg_param += 0.05
            if initial_reg_param < reg_parameter:
                n_iter = 10000
                total_iter_counter += n_iter
            else:
                n_iter = 100000 - total_iter_counter
            print("Training our objective with reg parameter " + str(initial_reg_param))

            for i in tqdm(range(n_iter)):
                optimizer_ours.zero_grad()
                # Draw samples from [-2, 2]^2
                X = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                our_nuc_loss, our_nuc_reg = our_nuc_reg_denoising_loss_function(h_model_our_nuc, g_model_our_nuc, target_fn, X, initial_reg_param)
                our_nuc_loss.backward()
                our_nuc_losses.append(our_nuc_loss.item())
                our_nuc_vals.append(our_nuc_reg.item())
                optimizer_ours.step()
                if i % 100 == 0:
                    print(i, our_nuc_loss.item())
                    # Compute average value of function on unit disc
                    n_samples = batch_size
                    X_disc = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                    X_disc = X_disc[torch.linalg.norm(X_disc, dim=1) <= 1]
                    y_disc_ours = g_model_our_nuc(h_model_our_nuc(X_disc)).squeeze().detach().cpu()
                    avg_val_ours = y_disc_ours.mean()
                    avg_values_our_nuc.append(avg_val_ours.item())
                    print("avg val: " + str(avg_val_ours.item()))
                    # Compute absolute error between ours and true sol function on unit square
                    X_square = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                    y_square_ours = g_model_our_nuc(h_model_our_nuc(X_square)).squeeze().detach().cpu()
                    y_square_target = (1 - d*reg_parameter) * target_fn(X_square).squeeze().detach().cpu()
                    abs_err = torch.mean(torch.abs(y_square_ours - y_square_target))
                    abs_errors_our_nuc.append(abs_err.item())
                    print("abs error: " + str(abs_err.item()))
                    # Compute average value of function outside the unit disc
                    X_outside = 20 * torch.rand(batch_size, base_dims, device=device) - 10
                    X_outside = X_outside[torch.linalg.norm(X_outside, dim=1) > 1]
                    y_outside_ours = g_model_our_nuc(h_model_our_nuc(X_outside)).squeeze().detach().cpu()
                    avg_val_outside_ours = y_outside_ours.mean()
                    outside_values_our_nuc.append(avg_val_outside_ours.item())
                    print("avg val outside: " + str(avg_val_outside_ours.item()))

    return h_model_our_nuc, g_model_our_nuc, our_nuc_losses, our_nuc_vals, avg_values_our_nuc, abs_errors_our_nuc

# Run experiments

def main():
    # Fix the Fourier map for all experiments
    n_freq = 500
    fourier_map = InputMapping(d_in=2, n_freq=n_freq, sigma=1, incrementalMask=False).to(device)

    reg_param_list = [0.10, 0.25]

    # Create directory to store results
    if not os.path.exists('/results/rof_ours_denoising_vs_exact_results_100k_iters'):
        os.makedirs('/results/rof_ours_denoising_vs_exact_results_100k_iters')

    for reg_param in reg_param_list:
        # Train exact objective
        model_exact_nuc, exact_nuc_losses, exact_nuc_vals, avg_values_exact_nuc, abs_errors_exact_nuc = train_exact_objective(target_fn, fourier_map, reg_param)
        # Save results
        torch.save(model_exact_nuc.state_dict(), '/results/rof_ours_denoising_vs_exact_results_100k_iters/exact_nuc_model_reg_param_' + str(reg_param) + '.pt')
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/exact_nuc_losses_reg_param_' + str(reg_param) + '.npy', exact_nuc_losses)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/exact_nuc_vals_reg_param_' + str(reg_param) + '.npy', exact_nuc_vals)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/avg_values_exact_nuc_reg_param_' + str(reg_param) + '.npy', avg_values_exact_nuc)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/abs_errors_exact_nuc_reg_param_' + str(reg_param) + '.npy', abs_errors_exact_nuc)

        # Train our objective
        h_model_our_nuc, g_model_our_nuc, our_nuc_losses, our_nuc_vals, avg_values_our_nuc, abs_errors_our_nuc = train_our_objective(target_fn, fourier_map, reg_param)
        # Save results
        torch.save(h_model_our_nuc.state_dict(), '/results/rof_ours_denoising_vs_exact_results_100k_iters/our_nuc_h_model_reg_param_' + str(reg_param) + '.pt')
        torch.save(g_model_our_nuc.state_dict(), '/results/rof_ours_denoising_vs_exact_results_100k_iters/our_nuc_g_model_reg_param_' + str(reg_param) + '.pt')
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/our_nuc_losses_reg_param_' + str(reg_param) + '.npy', our_nuc_losses)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/our_nuc_vals_reg_param_' + str(reg_param) + '.npy', our_nuc_vals)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/avg_values_our_nuc_reg_param_' + str(reg_param) + '.npy', avg_values_our_nuc)
        np.save('/results/rof_ours_denoising_vs_exact_results_100k_iters/abs_errors_our_nuc_reg_param_' + str(reg_param) + '.npy', abs_errors_our_nuc)

if __name__ == "__main__":
    main()