import numpy as np
import pickle
from cluster_SZ import pathfinder, cluster_info, healpix, power_spectrum
tmpdir_path=pathfinder.tmpdir_path()

def chains_path(amas):
    return tmpdir_path+'mean_models/'+amas+'_transfert_MCMC_chains.pkl'

def covar_path(amas):
    return tmpdir_path+'covar/'+amas+'_covar_transfert_3R500.npy'

def chains(amas):
    with open(chains_path(amas),'rb') as f:
        ch = pickle.load(f)
    return ch

def covar(amas):
    with open(covar_path(amas),'rb') as f:
        cov = np.load(f)
    return cov

def bestfit(amas):
    ch = chains(amas)
    bfit_dict = {
        'xc':np.median(ch['xc']),
        'yc':np.median(ch['yc']),
        'angle':np.median(ch['angle']),
        'ell':np.median(ch['ell']),
        'P0':np.median(ch['P0']),
        'c500':np.median(ch['c_500']),
        'beta':np.median(ch['beta'])
    }
    err_dict = {
        'xc': np.std(ch['xc']),
        'yc': np.std(ch['yc']),
        'angle': np.std(ch['angle']),
        'ell': np.std(ch['ell']),
        'P0': np.std(ch['P0']),
        'c500': np.std(ch['c_500']),
        'beta': np.std(ch['beta'])
    }

    return {
        'bfit':bfit_dict,
        'err':err_dict
    }

def chi2(amas):
    full_im = cluster_info.image(amas,10).flatten()
    mask = power_spectrum.mask_SZ(amas).flatten()
    offset = np.mean(np.delete(full_im,np.where(mask==0)))
    print('offset :',offset)
    true_im = cluster_info.image(amas,3)
    bfit_im = cluster_info.mm_bfit(amas,3)
    true_im_healpix = np.dot(healpix.transfert(amas,3).matrix,true_im.flatten())
    bfit_im_healpix = np.dot(healpix.transfert(amas,3).matrix,bfit_im.flatten()) + offset
    inv_cov = np.linalg.inv(covar(amas))
    return np.dot(bfit_im_healpix-true_im_healpix,np.dot(inv_cov,bfit_im_healpix-true_im_healpix))

AMAS = ['A85','A644','A1644','A1795','A2029','A2142','A2255','A2319','A3158','A3266','RXC1825','ZW1215']

for amas in AMAS:
    #print('chi2 '+amas+' : ',chi2(amas))
    stats = bestfit(amas)
    size = np.shape(covar(amas))[0]
    print(amas+' (nombre de pixels : '+str(size)+')')
    print('Bestfit '+amas+' : ',stats['bfit'])
    print('Erreur '+amas+' : ',stats['err'])
    print('-------------------------------')

