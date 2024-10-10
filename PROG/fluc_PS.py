import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pickle
from cluster_SZ import cluster_info, pathfinder
tmpdir_path=pathfinder.tmpdir_path()

amas='join'
n_R500_low=0
n_R500_high=2
filepath_ell=tmpdir_path+'fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500.pkl'
filepath_full=tmpdir_path+'fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_FULL.pkl'
filepath_noise=tmpdir_path+'fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_noise.pkl'

with open(filepath_full, 'rb') as f:
    chains_full = pickle.load(f)

with open(filepath_ell,'rb') as f:
    chains_ell = pickle.load(f)

with open(filepath_noise,'rb') as f:
    chains_noise = pickle.load(f)

chains_k_inj_ell = chains_ell[:,:,0].flatten()
chains_slope_ell = chains_ell[:,:,1].flatten()
chains_log_norm_ell = chains_ell[:,:,2].flatten()
chains_norm_ell = 10**chains_log_norm_ell

chains_k_inj_full = chains_full[:,:,0].flatten()
chains_slope_full = chains_full[:,:,1].flatten()
chains_log_norm_full = chains_full[:,:,2].flatten()
chains_norm_full = 10**chains_log_norm_full

chains_k_inj_noise = chains_noise[:,:,0].flatten()
chains_slope_noise = chains_noise[:,:,1].flatten()
chains_log_norm_noise = chains_noise[:,:,2].flatten()
chains_norm_noise = 10**chains_log_norm_noise

R = np.geomspace(0.1,1,500)
K = 1/cluster_info.R500_to_pix(R)

pow_spec_chains_ell = chains_norm_ell[None,...]*np.exp(-chains_k_inj_ell[None,...]/K[...,None])*K[...,None]**(-chains_slope_ell[None,...])

pow_spec_chains_full = chains_norm_full[None,...]*np.exp(-chains_k_inj_full[None,...]/K[...,None])*K[...,None]**(-chains_slope_full[None,...])

pow_spec_chains_noise = chains_norm_noise[None,...]*np.exp(-chains_k_inj_noise[None,...]/K[...,None])*K[...,None]**(-chains_slope_noise[None,...])

font = {
    'size':20,
    'weight':'normal'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

plt.fill_between(R,*np.percentile(pow_spec_chains_ell,[16,84],axis=1)*K**3,color='blue',alpha=0.5)
plt.fill_between(R,*np.percentile(pow_spec_chains_noise,[16,84],axis=1)*K**3,color='red',alpha=0.5)
plt.fill_between(R,*np.percentile(pow_spec_chains_full,[16,84],axis=1)*K**3,color='green',alpha=0.5)
plt.legend(['Amas elliptiques','Bruit','Tous les amas'],prop={'size':20},loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{500}$')
plt.ylabel(r'$P_{3D}(k) \times k^3$')
#plt.ylim([1e-5,2e-3])
plt.grid()

plt.show()

