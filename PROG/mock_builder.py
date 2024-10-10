import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from cluster_SZ import power_spectrum, mean_model, cluster_info, healpix

amas='A2142'
n_R500 = 10
angle = 1
ell = 0.5
P0 = 7.6
c_500 = 1.66
beta = 4.16

P500 = cluster_info.P500(amas)
theta500 = cluster_info.theta500(amas)
FWHM=np.array(mean_model.std_planck_arcmin*1024/(20*theta500))
xc_center,yc_center = jnp.array(cluster_info.center_coord(amas, n_R500))
spnorm = cluster_info.spnorm(n_R500)
knorm = cluster_info.knorm(n_R500)
spgrid = cluster_info.spgrid(n_R500)
PSF = mean_model.PSF_Planck(theta500,spnorm)
mock_mm = mean_model.convol_Mock_Y_map(P500,PSF,spgrid,xc_center,yc_center,angle,ell,P0,c_500,beta)

image=mock_mm

PSgrid = power_spectrum.PS_noise_amas(amas).PS_grid(n_R500)
white_noise = np.random.normal(size=np.shape(image))
noise = np.real(np.fft.ifft2(np.fft.fft2(white_noise)*np.sqrt(PSgrid)))
image=image+noise

image_rebin=np.zeros(np.shape(image))
PIXEL=healpix.pix_image(amas,n_R500)
for h in np.unique(PIXEL):
    mask = PIXEL == h
    pixel_mean = np.mean(image[mask])
    image_rebin[mask] = pixel_mean

transfo_mock_test = {
    'xc':xc_center,
    'yc':yc_center,
    'angle':angle,
    'ell':ell,
    'P0':P0,
    'c_500':c_500,
    'beta':beta,
    'mock_image':image_rebin
}

with open('mock_tests/transfo_mock_test.pkl','wb') as f:
    pickle.dump(transfo_mock_test,f)

plt.imshow(image_rebin)
plt.show()