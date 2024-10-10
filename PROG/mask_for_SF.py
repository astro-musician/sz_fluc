import numpy as np
from cluster_SZ import healpix, power_spectrum, cluster_info, pathfinder

amas='A2142'
n_R500 = 10
tmpdir_path = pathfinder.tmpdir_path()

mask_cluster=power_spectrum.mask_SZ(amas).flatten()
im_cluster=cluster_info.image(amas,n_R500).flatten()
PIXEL = healpix.pix_image(amas,n_R500).flatten()
unique_PIXEL = np.unique(PIXEL)
mask_healpix = np.zeros(len(unique_PIXEL))
im_healpix = np.zeros(len(unique_PIXEL))

for h in range(len(unique_PIXEL)):
    mask = PIXEL == unique_PIXEL[h]
    mask_healpix[h] = np.sum(mask_cluster[mask]) / np.sum(mask)
    im_healpix[h] = np.sum(im_cluster[mask]) / np.sum(mask)

mask_healpix=(mask_healpix==1)

with open(tmpdir_path+'power_spectra/'+amas+'_mask_healpix.npy','wb') as f:
    np.save(f,mask_healpix)

with open(tmpdir_path+'power_spectra/'+amas+'_healpix.npy','wb') as f:
    np.save(f,im_healpix)
