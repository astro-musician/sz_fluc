import torch
import jax.numpy as jnp
from cluster_SZ import healpix

with open('posteriors/simulations/A2142_fluc_map_simu_x.npy','rb') as f:
    x=jnp.load(f,allow_pickle=True)
x=x.numpy()
size_x, size_y = jnp.shape(x)[1], jnp.shape(x)[2]
x=x.reshape(100000,size_x*size_y)
print(jnp.shape(x))
transfert=healpix.transfert('A2142',2.5).matrix
transfert=transfert.T
print(jnp.shape(transfert))

healpix_x = jnp.matmul(x,transfert)

with open('posteriors/simulations/A2142_fluc_map_simu_x_healpix.npy','wb') as f:
    jnp.save(f,healpix_x)
