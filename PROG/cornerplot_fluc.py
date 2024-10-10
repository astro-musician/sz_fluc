import numpy as np
import pickle
from chainconsumer import Chain, ChainConsumer, ChainConfig, PlotConfig, Truth
import pandas as pd
from scipy.stats import binned_statistic
import arviz as az
import matplotlib.pyplot as plt
from cluster_SZ import cluster_info, pathfinder
tmpdir_path=pathfinder.tmpdir_path()

amas='join'
n_R500_low=0
n_R500_high=2
filepath=tmpdir_path+'fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_noise.pkl'

c=ChainConsumer()

with open(filepath, 'rb') as f:
    chains = pickle.load(f)

n_iter, n_walkers, n_dim = np.shape(chains)

chains_k_inj = chains[:,:,0].flatten()
chains_slope = chains[:,:,1].flatten()
chains_log_norm = chains[:,:,2].flatten()
chains_l_inj = cluster_info.pix_to_R500(2*np.pi/chains_k_inj) # En R500
chains_norm = 10**chains_log_norm

K = np.geomspace(2*np.pi/cluster_info.R500_to_pix(3),2*np.pi/cluster_info.R500_to_pix(0.1),100)
pow_spec_chains = chains_norm[None,...]*np.exp(-chains_k_inj[None,...]/K[...,None])*K[...,None]**(-chains_slope[None,...])
med_pow_spec = np.median(pow_spec_chains,axis=1)
std_pow_spec = np.std(pow_spec_chains,axis=1)/np.sqrt(np.shape(pow_spec_chains)[1])

chains_kpeak = chains_k_inj/(chains_slope-3)
chains_mach = 2.4*np.sqrt(4*np.pi*chains_kpeak**3*chains_norm*np.exp(-chains_k_inj/chains_kpeak)*chains_kpeak**(-chains_slope))
gamma = 5/3 # indice polytropique
chains_bias = chains_mach**2*gamma/(chains_mach**2*gamma+3)

chaindict = {
    r'$l_{inj} \: (R_{500})$':chains_l_inj,
    r'slope':chains_slope,
    r'Mach':chains_mach
}

font = {
    'size':25,
    'weight':'semibold'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

df = pd.DataFrame(data=chaindict)
c.add_chain(Chain(samples=df,name=amas,show_contour_labels=True,marker_style='*'))
c.add_truth(Truth(location={'slope':11/3})) # Pente de Kolmogorov
c.set_plot_config(PlotConfig(label_font_size=20,tick_font_size=12,summary_font_size=20))
fig=c.plotter.plot()

plt.figure(2)
plt.plot(K*1024/20,med_pow_spec*(20/1024)**3,'-b')
plt.fill_between(K*1024/20,(med_pow_spec-std_pow_spec)*(20/1024)**3,(med_pow_spec+std_pow_spec)*(20/1024)**3,color='cornflowerblue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{500}^{-1}$')
plt.ylabel(r'$P_{3D}$')
plt.grid()
plt.title('Zone de calcul du bruit')

plt.show()

