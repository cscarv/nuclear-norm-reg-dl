import torch
from functorch import vmap, vjp, jvp

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Compute Jacobian-vector product of Jmodel(x) with w
def model_jvp(model, x, w):
    return jvp(model, (x,), (w,))[1]

# Compute Jacobian-vector product of Jmodel(x) with W using vmap
def model_jvp_vmap(model, x, W):
    fixed_model_jvp = lambda x, w: model_jvp(model, x, w)
    return vmap(fixed_model_jvp, in_dims=(None,1), out_dims=1)(x, W)

# Compute vector-Jacobian product of v with Jmodel(x) using vjp
def model_vjp(model, x, v):
    vjp_func = vjp(model, x)[1]
    return vjp_func(v)[0]

# Compute vector-Jacobian product of M with Jmodel(x) using vmap
def model_vjp_vmap(model, x, M):
    fixed_model_vjp = lambda m: model_vjp(model, x, m)
    return vmap(fixed_model_vjp)(M)

# Randomized range finder
def randomized_range(model, x, rank, oversampling_factor=10):
    """Randomized range finder for the Jacobian of a model at x.
    Based on Algorithm 4.1 of Halko et al. (2009)."""
    in_dims = x.shape[0]
    Y = torch.randn(in_dims, rank + oversampling_factor, device=device) # (in_dims, rank + oversampling_factor)
    JtimesY = model_jvp_vmap(model, x, Y).detach() # (out_dims, rank + oversampling_factor)
    Q, _ = torch.linalg.qr(JtimesY, mode='reduced') # (out_dims, rank + oversampling_factor)
    return Q

# Randomized SVD of the Jacobian
def randomized_svd(model, x, rank, oversampling_factor=10):
    """Randomized SVD for the Jacobian of a model at x.
    Based on Algorithm 4.1 of Halko et al. (2009)."""
    Q = randomized_range(model, x, rank, oversampling_factor) # (out_dims, rank+oversampling_factor)
    B = model_vjp_vmap(model, x, Q.T).detach() # (rank+oversampling_factor, in_dims)
    U, S, V = torch.linalg.svd(B, full_matrices=False) # (rank+oversampling_factor, rank+oversampling_factor), (rank+oversampling_factor,), (rank+oversampling_factor, in_dims)
    U = U[:, :rank] # (rank+oversampling_factor, rank)
    S = S[:rank] # (rank,)
    V = V[:rank, :] # (rank, in_dims)
    return Q @ U, S, V # (out_dims, rank), (rank,), (rank, in_dims)

# Randomized SVD of Jacobian evaluated at a batch of inputs

def batch_randomized_svd(model, X, rank, oversampling_factor=10):
    U, S, V = vmap(randomized_svd, in_dims=(None, 0, None, None), out_dims=0, randomness='different')(model, X, rank, oversampling_factor).detach()
    return U, S, V # (batch_size, out_dims, rank), (batch_size, rank), (batch_size, rank, in_dims)