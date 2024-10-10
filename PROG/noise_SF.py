import numpy as np
import matplotlib.pyplot as plt
import pickle
from cluster_SZ import cluster_info, SF

amas='A2142'
n_R500_low=0
n_R500_high=2
n_bins=20

with open('power_spectra/'+amas+'_mask_healpix.npy','rb') as f:
    mask_healpix = np.load(f)

with open('power_spectra/'+amas+'_healpix.npy','rb') as f:
    im_healpix = np.load(f)

bins = np.geomspace(2*cluster_info.healpix_mean_side(amas),cluster_info.R500_to_pix(n_R500_high-n_R500_low),n_bins+1)
lows=bins[:-1]
highs=bins[1:]

X = cluster_info.pix_to_R500(lows+highs)/2
X_err = cluster_info.pix_to_R500(highs-lows)/2
Y = SF.structure_function(im_healpix,amas,10,lows,highs,mask_healpix)
Y_err = SF.structure_function_err(im_healpix,amas,10,lows,highs,mask_healpix)/2

info={
    'X':X,
    'X_err':X_err,
    'Y':Y,
    'Y_err':Y_err
}

dictpath='SF/'+amas+'_noise_SF.pkl'
with open(dictpath,'wb') as f:
    pickle.dump(info,f)