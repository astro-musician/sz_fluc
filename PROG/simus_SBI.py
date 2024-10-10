import torch

from sbi import utils as utils
from sbi.inference import SNLE, SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

import numpy as np
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import jax.random as random
from jax import jit
import pickle
from cluster_SZ import mean_model, power_spectrum, pathfinder, cluster_info, mock_fluc

import matplotlib.pyplot as plt

amas='A2142'
n_R500=2
num_simulations=100000
space_sampling=5 # depend de l'amas

n_dim=3

def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

# Choix du masque et de l'intervalle de calcul du power spectrum pour les simulations

mask = jnp.ones(cluster_info.pix_shape(n_R500))
k_min = 1/cluster_info.R500_to_pix(n_R500)
k_max = 1/cluster_info.R500_to_pix(0.05)

# Bornes des distributions uniformes pour les paramètres de turbulence (Injection en pixels**-1, pente logarithmique et log de la norme)

low_inj = float(1/cluster_info.R500_to_pix(5))
high_inj = float(1/cluster_info.R500_to_pix(0.1))
low_slope = 1
high_slope = 5
low_log_norm = -3
high_log_norm = 0

prior=utils.BoxUniform(low=torch.Tensor([low_inj,low_slope,low_log_norm]),high=torch.Tensor([high_inj,high_slope,high_log_norm]))
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Importation des fonctions nécessaires

spnorm=cluster_info.spnorm(n_R500)
knorm=cluster_info.knorm(n_R500)
cube_size=cluster_info.pix_shape(n_R500)[0]
P500 = cluster_info.P500(amas)
theta500 = cluster_info.theta500(amas)
spgrid = cluster_info.spgrid(n_R500)
spgrid_3D = cluster_info.spgrid_3D(n_R500)
knorm_3D = cluster_info.knorm_3D(n_R500)
L_size=np.shape(knorm_3D)[2]
xc,yc = cluster_info.center_coord(amas, n_R500)
PSF = mean_model.PSF_Planck(theta500,spnorm)

cube_size_rebin=np.shape(cluster_info.rebin(cluster_info.image(amas,n_R500),space_sampling))[0]
spnorm_rebin=cluster_info.spnorm_rebin(n_R500,space_sampling)
spgrid_rebin=cluster_info.spgrid_rebin(n_R500,space_sampling)
spgrid_3D_rebin=[cluster_info.rebin_3D(spgrid_3D[0],space_sampling),cluster_info.rebin_3D(spgrid_3D[1],space_sampling),cluster_info.rebin_3D(spgrid_3D[2],space_sampling)]
knorm_3D_rebin=cluster_info.rebin_3D(knorm_3D,space_sampling)
PSF_rebin=cluster_info.rebin(PSF,space_sampling)*space_sampling**2 # La PSF doit être normalisée en fonction du rebin

with open('transfert/'+amas+'_transfo_spgrid.pkl','rb') as f:
    transfo_dict=pickle.load(f)
    transfo_spgrid=transfo_dict['transfo_spgrid']
    transfo_det=transfo_dict['det']

transfo_center=transfo_dict['center']
sursamp=transfo_dict['sursamp']

transfo_spnorm = np.sqrt((transfo_spgrid[0]-transfo_center[0])**2+(transfo_spgrid[1]-transfo_center[1])**2)#cluster_info.spnorm(10)[np.intc(transfo_spgrid[0]),np.intc(transfo_spgrid[1])]
PSF_transfo = transfo_det * mean_model.PSF_Planck(theta500,transfo_spnorm)
spgrid_3D_transfo = cluster_info.spgrid_3D_transfo(amas)
knorm_3D_transfo = cluster_info.knorm_3D_transfo(amas)
cube_size_transfo = jnp.shape(knorm_3D_transfo)[0]

# Modèle moyen (best fit)
n_R500_bestfit=10
prog_path=pathfinder.prog_path()
filepath = 'mean_models/' + amas + '_transfo_MCMC_chains_2x2.pkl'
with open(filepath, 'rb') as f:
    chains = pickle.load(f)

chains_xc = chains['xc'].flatten()
chains_yc = chains['yc'].flatten()
# Recentrer les positions xc yc en sachant que le modèle moyen est fit à 2R500 hors transfo !
#chains_yc+= cluster_info.R500_to_pix(n_R500-n_R500_bestfit)
#chains_xc+= cluster_info.R500_to_pix(n_R500-n_R500_bestfit)

chains_angle = chains['angle'].flatten()
chains_ell = chains['ell'].flatten()
chains_P0 = chains['P0'].flatten()
chains_c500 = chains['c_500'].flatten()
chains_beta = chains['beta'].flatten()

num_samples = len(chains_beta)
indexing_mean_model = np.arange(num_samples)
#mean_model_2D = mean_model.convol_Mock_Y_map(P500,PSF_rebin,spgrid_rebin,np.median(chains_xc),np.median(chains_yc),np.median(chains_angle),np.median(chains_ell),np.median(chains_P0),np.median(chains_c500),np.median(chains_beta))
mean_model_2D = mean_model.convol_Mock_Y_map(P500,PSF_transfo,transfo_spgrid,np.median(chains_xc),np.median(chains_yc),np.median(chains_angle),np.median(chains_ell),np.median(chains_P0),np.median(chains_c500),np.median(chains_beta))

PSgrid = power_spectrum.PS_noise_amas(amas).PS_grid(n_R500,savefile=False)
PSgrid_rebin=cluster_info.rebin(PSgrid,round(cluster_info.healpix_mean_side(amas)))
PSgrid_transfo = power_spectrum.PS_noise_amas(amas).PS_grid_transfo(savefile=True)/transfo_det
print('Power spectrum grid loaded')

L=np.linspace(-5,5,L_size) # Array pour intégrer en échelle R500 sur la ligne de visée (jusqu'à 5 R500 de chaque côté)

# POUR RECOUVRER LA VERSION SANS TRANSFORMATION, REMPLACER LES GRILLES SPATIALE ET SPECTRALE PAR LEURS VERSIONS NON TRANSFORMÉES
def mock_image_jax(xc,yc,angle,ell,P0,c_500,beta,kcut,slope,log_norm):

    # CUBE : AMAS ET FLUCTUATIONS
    mock_mean_cube = mean_model.mean_model_3D(P500, spgrid_3D_transfo, cube_size_transfo, L_size, xc, yc, angle, ell, P0, c_500,beta).cube  # Modèle moyen 3D
    mock_turb_cube = mock_fluc.mock_fluc_cube(cube_size_transfo, L_size, knorm_3D_transfo, kcut, slope,10 ** log_norm)  # Cube de fluctuations turbulentes (relatives, delta_P/P)
    mock_fluc_cube =  mock_mean_cube * ( jnp.ones((cube_size_transfo,cube_size_transfo,L_size)) + mock_turb_cube )  # Cube d'amas avec fluctuations Y

    # IMAGE (intégration + PSF)
    mock_image = jsi.trapezoid(mock_fluc_cube,axis=2,x=L)  # Image des fluctuations relatives (intégration le long de la ligne de visée)
    mock_image = jnp.real(jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fft2(mock_image) * jnp.fft.fft2(PSF_transfo))))

    # BRUIT
    white_noise = random.normal(rng_key(), shape=jnp.shape(PSgrid_transfo))
    noise = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(white_noise) * jnp.sqrt(PSgrid_transfo)))

    # IMAGE COMPLÈTE
    mock_image = mock_image + noise

    return mock_image - mean_model_2D

mock_image_jit = jit(mock_image_jax)

def mock_image(theta):
    kcut, slope, log_norm = theta.numpy()

    index_mm = np.random.choice(indexing_mean_model)
    xc, yc, angle, ell, P0, c_500, beta = chains_xc[index_mm], chains_yc[index_mm], chains_angle[index_mm], chains_ell[index_mm], chains_P0[index_mm], chains_c500[index_mm], chains_beta[index_mm]

    mock_fluc_image = mock_image_jit(xc,yc,angle,ell,P0,c_500,beta,kcut,slope,log_norm)

    #return torch.from_numpy(np.array(power_spectrum.power_spectrum(mock_fluc_image,spnorm,mask=mask,k_min=k_min,k_max=k_max)[1]))
    return torch.Tensor(np.array(mock_fluc_image))

simulator = process_simulator(mock_image,prior,prior_returns_numpy)
check_sbi_inputs(simulator,prior)
theta, x = simulate_for_sbi(simulator,proposal=prior,num_simulations=num_simulations)
#x_PS = power_spectrum.power_spectrum_mock_fluc(x.numpy(),transfo_spnorm,jnp.ones(jnp.shape(transfo_spnorm)),1/cluster_info.R500_to_pix(2),1/cluster_info.R500_to_pix(0.1),theta.numpy()[:,2])[0]
#x_PS = torch.from_numpy(np.array(x_PS))

#inference=SNPE(prior=prior)
#inference=inference.append_simulations(theta,x_PS)
#density_estimator=inference.train()
#posterior=inference.build_posterior(density_estimator)
#print(posterior)


with open('posteriors/simulations/'+amas+'_fluc_map_simu_theta.pkl','wb') as f:
    pickle.dump(theta,f)

#with open('posteriors/simulations/'+amas+'_fluc_map_simu_x.pkl','wb') as f:
#    pickle.dump(x,f)
