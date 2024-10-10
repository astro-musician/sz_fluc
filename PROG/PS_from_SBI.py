import torch
import jax.numpy as jnp
import numpy as np
import pickle
import os
from cluster_SZ import cluster_info, power_spectrum

amas = 'A85'
n_R500 = 2.5
n_R500_mask_high = 2
n_R500_mask_low = 0

spnorm = cluster_info.spnorm(n_R500)
mask = (spnorm>cluster_info.R500_to_pix(n_R500_mask_low))&(spnorm<cluster_info.R500_to_pix(n_R500_mask_high))

with open('posteriors/simulations/'+amas+'_fluc_map_simu_theta_TEST.pkl','rb') as f:
    theta = pickle.load(f)

with open('posteriors/simulations/'+amas+'_fluc_map_simu_x_TEST.pkl','rb') as f:
    x = pickle.load(f)

slopes = theta.numpy()[:,1]

k_min=1/cluster_info.R500_to_pix(2)
k_max=1/cluster_info.R500_to_pix(0.1)

PS = power_spectrum.power_spectrum_mock_fluc(x.numpy(),spnorm,mask,k_min,k_max,slopes)
print('num_PS, num_rings : ',jnp.shape(PS[0]))

PS=torch.Tensor(np.array(PS[0]))

with open('posteriors/simulations/'+amas+'_'+str(n_R500_mask_low)+'_to_'+str(n_R500_mask_high)+'R500_mock_power_spectra.pkl','wb') as f:
    pickle.dump(PS,f)

