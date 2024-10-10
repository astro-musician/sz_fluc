n_walkers=4

import numpyro
numpyro.set_host_device_count(n_walkers)
from cluster_SZ import mean_model, healpix, pathfinder, cluster_info
import numpyro.distributions as dist
from numpyro.infer import NUTS, BarkerMH, MCMC
from numpyro.infer.initialization import init_to_mean, init_to_value
import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax.lax import dot
from jax import jit
from astropy.io import fits
import pickle

import matplotlib.pyplot as plt

run_mcmc=True
amas='A2255'
n_R500=3
rebin_method='transfert'
space_sampling= 4
init = {'xc':cluster_info.pix_shape(n_R500)[0]/2,
		'yc':cluster_info.pix_shape(n_R500)[0]/2,
		'angle':0,
		'ell':0.5,
		'P0':7.0,
		'c_500':1.14,
		'beta':4.0}

n_train=1000
n_iter=1000

prog_path=pathfinder.prog_path()

imagepath=prog_path+'../data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
image=cluster_info.reduce_n_R500(jnp.float32(fits.open(imagepath)[0].data),n_R500)
def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

key_mcmc=random.split(rng_key(),n_walkers)

with open("covar/"+amas+"_covar_"+rebin_method+"_"+str(n_R500)+"R500.npy",'rb') as f:
	covar=jnp.load(f)

print('covariance matrix loaded')

with open('transfert/'+amas+'_transfo_spgrid.pkl','rb') as f:
    transfo_dict=pickle.load(f)
    transfo_spgrid=transfo_dict['transfo_spgrid']
    transfo_det=transfo_dict['det']

transfo_center=transfo_dict['center']
sursamp=transfo_dict['sursamp']

if rebin_method=='transfert':
	transfert=healpix.transfert(amas,n_R500,savefile=False).matrix
	print('Transfert matrix loaded')
	image_healpix=dot(transfert,image.flatten())

elif rebin_method=='indices':
	indices_healpix_x, indices_healpix_y = healpix.healpix_index(amas,n_R500,savefile=True).indices
	print('Healpix indices loaded')
	image_healpix=image[indices_healpix_x,indices_healpix_y]

elif rebin_method=='rebin':
	image_healpix=cluster_info.rebin(image,space_sampling).flatten()

elif rebin_method=='transfo':
	image_transfo=cluster_info.image(amas,10)[np.intc(transfo_spgrid[0]),np.intc(transfo_spgrid[1])]
	image_transfo=cluster_info.rebin(image_transfo,sursamp)
#	with open('mock_tests/transfo_mock_test.pkl','rb') as f:
#		image_test = pickle.load(f)['mock_image']
#	image_transfo = cluster_info.rebin( image_test[np.intc(transfo_spgrid[0]), np.intc(transfo_spgrid[1])] , sursamp)
	image_healpix=image_transfo.flatten()

else:
	print('Method can only be transfert, indices or rebin.')
	run_mcmc=False

P500 = cluster_info.P500(amas)
theta500 = cluster_info.theta500(amas)
FWHM=jnp.array(mean_model.std_planck_arcmin*1024/(20*theta500))
xc_center,yc_center = jnp.array(cluster_info.center_coord(amas, n_R500))
spnorm = cluster_info.spnorm(n_R500)
knorm = cluster_info.knorm(n_R500)
spgrid = cluster_info.spgrid(n_R500)
PSF = mean_model.PSF_Planck(theta500,spnorm)

spnorm_rebin=cluster_info.spnorm_rebin(n_R500,space_sampling)
spgrid_rebin=cluster_info.spgrid_rebin(n_R500,space_sampling)
PSF_rebin=cluster_info.rebin(PSF,space_sampling)*space_sampling**2 # La PSF doit être normalisée en fonction du rebin
xc_rebin=xc_center/space_sampling
yc_rebin=yc_center/space_sampling
FWHM_rebin=FWHM/space_sampling

transfo_spnorm = np.sqrt((transfo_spgrid[0]-transfo_center[0])**2+(transfo_spgrid[1]-transfo_center[1])**2)#cluster_info.spnorm(10)[np.intc(transfo_spgrid[0]),np.intc(transfo_spgrid[1])]
PSF_transfo = transfo_det * mean_model.PSF_Planck(theta500,transfo_spnorm)

mock_mean_model = jit(mean_model.convol_Mock_Y_map)
angle_app=init['angle']
def model_rebin():
	angle = numpyro.sample('angle', dist.Uniform(angle_app-jnp.pi/2, angle_app+jnp.pi/2)) # Centrer sur la valeur apparente
	ell = numpyro.sample('ell', dist.Uniform(0, 0.99))
	P0 = numpyro.sample('P0', dist.Uniform(0, 20))
	c_500 = numpyro.sample('c_500', dist.Uniform(0, 5))
	beta = numpyro.sample('beta', dist.Uniform(1, 15))
	#xc = numpyro.sample('xc', dist.Uniform(xc_center - FWHM, xc_center + FWHM))
	#yc = numpyro.sample('yc', dist.Uniform(yc_center - FWHM, yc_center + FWHM))
	spread_xc = numpyro.sample('spread_xc', dist.Uniform(-1, 1))
	spread_yc = numpyro.sample('spread_yc', dist.Uniform(-1, 1))
	xc = numpyro.deterministic('xc', xc_center + FWHM * spread_xc)
	yc = numpyro.deterministic('yc', yc_center + FWHM * spread_yc)

	mock_map = mock_mean_model(P500, PSF_rebin, spgrid_rebin, xc, yc, angle, ell, P0, c_500, beta).flatten()

	numpyro.sample('likelihood', dist.MultivariateNormal(loc=mock_map, covariance_matrix=covar),obs=image_healpix)


def model_transfert():

	angle=numpyro.sample('angle',dist.Uniform(init['angle']-jnp.pi/2,init['angle']+jnp.pi/2))
	ell=numpyro.sample('ell',dist.Uniform(0,0.99))
	P0=numpyro.sample('P0',dist.Uniform(1,20))
	c_500=numpyro.sample('c_500',dist.Uniform(0,5))
	beta=numpyro.sample('beta',dist.Uniform(1,15))
	xc = numpyro.sample('xc',dist.Uniform(xc_center-2*FWHM,xc_center+2*FWHM))
	yc = numpyro.sample('yc',dist.Uniform(yc_center-2*FWHM,yc_center+2*FWHM))
	
	mock_map=mock_mean_model(P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta)
	mock_map_healpix=dot(transfert,mock_map.flatten())

	numpyro.sample('likelihood',dist.MultivariateNormal(loc=mock_map_healpix,covariance_matrix=covar),obs=image_healpix)

def model_indices():

	angle = numpyro.sample('angle', dist.Uniform(-np.pi / 2, np.pi / 2))
	ell = numpyro.sample('ell', dist.Uniform(0, 0.99))
	P0 = numpyro.sample('P0', dist.Uniform(1, 12))
	c_500 = numpyro.sample('c_500', dist.Uniform(0,5))
	beta = numpyro.sample('beta', dist.Uniform(1, 10))
	#xc = numpyro.sample('xc', dist.Normal(xc_center, FWHM))
	#yc = numpyro.sample('yc', dist.Normal(yc_center, FWHM))
	spread_xc = numpyro.sample('spread_xc', dist.Normal(0, 1))
	spread_yc = numpyro.sample('spread_yc', dist.Normal(0, 1))
	xc = numpyro.deterministic('xc', xc_center + FWHM * spread_xc)
	yc = numpyro.deterministic('yc', yc_center + FWHM * spread_yc)

	mock_map = mean_model.mean_model(P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta).mean_model_map
	mock_map_healpix = mock_map[indices_healpix_x,indices_healpix_y]
	numpyro.sample('likelihood', dist.MultivariateNormal(loc=mock_map_healpix, covariance_matrix=covar),obs=image_healpix)

def model_transfo():
	angle = numpyro.sample('angle', dist.Uniform(angle_app -jnp.pi / 2, angle_app + jnp.pi / 2))
	ell = numpyro.sample('ell', dist.Uniform(0, 0.99))
	P0 = numpyro.sample('P0', dist.Uniform(1, 20))
	c_500 = numpyro.sample('c_500', dist.Uniform(0, 5))
	beta = numpyro.sample('beta', dist.Uniform(1, 15))
	xc = numpyro.sample('xc', dist.Uniform(xc_center-2*FWHM, xc_center+2*FWHM))
	yc = numpyro.sample('yc', dist.Uniform(yc_center-2*FWHM, yc_center+2*FWHM))
	#xc = numpyro.deterministic('xc', xc_center + spread_xc)
	#yc = numpyro.deterministic('yc', yc_center + spread_yc)

	mock_map_healpix= cluster_info.rebin(mean_model.convol_Mock_Y_map(P500,PSF_transfo,transfo_spgrid,xc,yc,angle,ell,P0,c_500,beta),sursamp ).flatten()
	numpyro.sample('likelihood', dist.MultivariateNormal(loc=mock_map_healpix, covariance_matrix=covar),obs=image_healpix)

if rebin_method=='transfert':
	kernel=NUTS(model_transfert,init_strategy=init_to_value(values=init))

elif rebin_method=='indices':
	kernel = NUTS(model_indices,init_strategy=init_to_value(values=init))

elif rebin_method=='rebin':
	kernel = NUTS(model_rebin,init_strategy=init_to_value(values=init))

elif rebin_method=='transfo':
	kernel = NUTS(model_transfo,init_strategy=init_to_value(values=init))

mcmc = MCMC(kernel,num_warmup=n_train,num_samples=n_iter,num_chains=n_walkers,progress_bar=True)

run_mcmc=True
if run_mcmc:
	
	mcmc.run(rng_key())
	chains=mcmc.get_samples(group_by_chain=True)
	with open('mean_models/'+amas+'_'+rebin_method+'_MCMC_chains.pkl','wb') as f:
		pickle.dump(chains,f)
		
