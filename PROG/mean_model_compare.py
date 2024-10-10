import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
from cluster_SZ import mean_model, pathfinder, cluster_info
tmpdir_path=pathfinder.tmpdir_path()

amas='RXC1825'
n_R500=10
n_R500_bestfit=3

savefits=True # Ne l'activer qu'avec n_R500 = 10 !

rebin_method='transfert'
filepath = tmpdir_path+'mean_models/'+amas+'_'+rebin_method+'_MCMC_chains.pkl'
image = cluster_info.image(amas,n_R500)

P500 = cluster_info.P500(amas)
theta500 = cluster_info.theta500(amas)
spnorm = cluster_info.spnorm(n_R500)
knorm = cluster_info.knorm(n_R500)
spgrid = cluster_info.spgrid(n_R500)
PSF = mean_model.PSF_Planck(theta500,spnorm)

with open(filepath,'rb') as f:
    chains = pickle.load(f)
    
angle = chains['angle'].flatten()
ell = chains['ell'].flatten()
xc = chains['xc'].flatten() + cluster_info.R500_to_pix(n_R500-n_R500_bestfit)
yc = chains['yc'].flatten() + cluster_info.R500_to_pix(n_R500-n_R500_bestfit)
P0 = chains['P0'].flatten()
c500 = chains['c_500'].flatten()
beta = chains['beta'].flatten()

print(np.median(xc),np.median(yc),np.median(angle),np.median(ell),np.median(P0),np.median(c500),np.median(beta))

mm_bfit = mean_model.mean_model(P500,PSF,spgrid,np.median(xc),np.median(yc),np.median(angle),np.median(ell),np.median(P0),np.median(c500),np.median(beta)).healpix(amas,n_R500,savefile=True)

if savefits:
    # Sauvegarde du fits

    hdu_mm=fits.PrimaryHDU(data=mm_bfit)
    new_header_mm=cluster_info.header(amas)
    hdu_mm.header=new_header_mm
    hdul_mm=fits.HDUList([hdu_mm])
    hdul_mm.writeto(tmpdir_path+'mean_models/'+amas+'_bestfit.fits',overwrite=True)

    # Sauvegarde de la carte de fluctuations

    fluc=cluster_info.image(amas,10) - mm_bfit
    hdu_fluc=fits.PrimaryHDU(data=fluc)
    new_header_fluc=cluster_info.header(amas)
    hdu_fluc.header=new_header_fluc
    hdul_fluc = fits.HDUList([hdu_fluc])
    hdul_fluc.writeto(tmpdir_path+'fluc_maps/'+amas+'_fluc_map.fits', overwrite=True)

fig, axes = plt.subplots(ncols=2,nrows=1)
vmin=np.min(image)
vmax=np.max(image)

im=axes.flat[0].imshow(mm_bfit,vmin=vmin,vmax=vmax,interpolation=None)
axes.flat[1].imshow(image,vmin=vmin,vmax=vmax,interpolation=None)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.title(amas)

plt.show()