from cluster_SZ import mean_model_beta, healpix, pathfinder, cluster_info
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.lax import dot
from jax import jit
from astropy.io import fits

run_mcmc=True
amas='A2319'
n_R500=2
rebin_method='transfert'
n_walkers=1
n_train=1000
n_iter=1000

prog_path=pathfinder.prog_path()

imagepath=prog_path+'../data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
image=cluster_info.reduce_n_R500(jnp.float32(fits.open(imagepath)[0].data),n_R500)

if rebin_method=='transfert':
	transfert=healpix.transfert(amas,n_R500,savefile=False).matrix
	print('Transfert matrix loaded')
	image_healpix=dot(transfert,image.flatten())
	n_healpix = np.shape(transfert)[0]

elif rebin_method=='indices':
	indices_healpix_x, indices_healpix_y = healpix.healpix_index(amas,n_R500,savefile=True).indices
	print('Healpix indices loaded')
	image_healpix=image[indices_healpix_x,indices_healpix_y]
	n_healpix=len(indices_healpix_x)

else:
	print('Method can only be transfert or indices')
	run_mcmc=False

P500 = cluster_info.P500(amas)
theta500 = cluster_info.theta500(amas)
FWHM=mean_model_beta.std_planck_arcmin*1024/(20*theta500)
xc_center,yc_center = cluster_info.center_coord(amas, n_R500)
spnorm = cluster_info.spnorm(n_R500)
knorm = cluster_info.knorm(n_R500)
spgrid = cluster_info.spgrid(n_R500)
PSF = mean_model_beta.PSF_Planck(theta500,spnorm)

#xc=xc_center
#yc=yc_center

mock_mean_model = jit(mean_model_beta.convol_Mock_Y_map)
def mock_mm_healpix(h,xc, yc, angle, ell, P0, c_500, beta):
	mock_map = mock_mean_model(P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta)
	return dot(transfert,mock_map.flatten())[np.intc(h)]

H = np.arange(n_healpix)
P0_0 = np.linspace(2,10,2)
c_500_0 = np.linspace(0.5,3,2)
beta_0 = np.linspace(2,10,2)

for P0 in P0_0:
	for c_500 in c_500_0:
		for beta in beta_0:

			bestfit = curve_fit(mock_mm_healpix,H,image_healpix,p0=[xc_center,yc_center,0,0.5,P0,c_500,beta])
			print('curvefit start [P0,c_500,beta]=',[P0,c_500,beta],' : ',bestfit[0])

		
