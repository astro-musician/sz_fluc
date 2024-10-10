import torch

import numpy as np
import pickle
from cluster_SZ import cluster_info, power_spectrum, healpix, SF
import emcee

amas='A2319'
n_R500_low = 0
n_R500_high = 2
noise=True

postpath = 'posteriors/'+amas+'_fluc_likelihood_'+str(n_R500_low)+'_to_'+str(n_R500_high)+'R500.pkl'

with open(postpath,'rb') as f:
    posterior=pickle.load(f)

with open('SF/'+amas+'_fluc_SF.npy','rb') as f:
    sf=np.load(f)
    n_bins=len(sf)

    if noise:

        xc, yc = 300, 300
        n = 2
        im_cluster = cluster_info.portion(cluster_info.image(amas,10),xc,yc,n).flatten()
        mask_cluster = cluster_info.portion(power_spectrum.mask_SZ(amas),xc,yc,n).flatten()
        PIXEL = cluster_info.portion(healpix.pix_image(amas, 10),xc,yc,n).flatten()
        unique_PIXEL = np.unique(PIXEL)
        mask_healpix = np.zeros(len(unique_PIXEL))
        im_healpix = np.zeros(len(unique_PIXEL))

        for h in range(len(unique_PIXEL)):
            mask = PIXEL == unique_PIXEL[h]
            mask_healpix[h] = np.sum(mask_cluster[mask]) / np.sum(mask)
            im_healpix[h] = np.sum(im_cluster[mask]) / np.sum(mask)

        bins = np.geomspace(2 * cluster_info.healpix_mean_side(amas), cluster_info.R500_to_pix(n_R500_high - n_R500_low), n_bins + 1)
        lows = bins[:-1]
        highs = bins[1:]

        sf = SF.structure_function_noise(im_healpix, amas, xc, yc, n, lows, highs, mask_healpix)

x_obs = torch.from_numpy(np.array(sf))

def potential(theta):
    theta = torch.Tensor(np.array(theta))
    pot = posterior.potential(x=x_obs,theta=theta)
    return pot

n_walkers = 16
n_dim = 3
init = np.array([float(1/cluster_info.R500_to_pix(1)),11/3,-3])
scatter_kinj = 0.1
scatter_slope = 0.05
scatter_log_norm = 1

scatter = np.array([scatter_kinj,scatter_slope,scatter_log_norm])[None,...]*np.random.rand(n_walkers)[...,None]
p_init = init[None,...]*np.ones((n_walkers,n_dim)) + scatter

sampler = emcee.EnsembleSampler(n_walkers,n_dim,potential)
n_train = 5000
n_iter = 5000
training_state = sampler.run_mcmc(p_init, n_train,progress=True)
sampler.reset()
sampler.run_mcmc(training_state, n_iter,progress=True)

samples = sampler.get_chain()
if noise:
    samp_path = 'fluc/' + amas + '_chains_' + str(n_R500_low) + '-' + str(n_R500_high) + 'R500_noise.pkl'

else:
    samp_path = 'fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500.pkl'
with open(samp_path,'wb') as f:
    pickle.dump(samples,f)
