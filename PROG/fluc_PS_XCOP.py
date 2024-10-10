import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import matplotlib.pyplot as plt
import arviz as az
import pickle
from cluster_SZ import cluster_info, pathfinder
tmpdir_path=pathfinder.tmpdir_path()

gamma=5/3

# XMM (code Simon)

class KolmogorovPowerSpectrum(hk.Module):
    """
    Kolmogorov power spectrum
    """

    def __init__(self):
        super(KolmogorovPowerSpectrum, self).__init__()

    def __call__(self, k):
        log_sigma = hk.get_parameter("log_sigma", [], init=-1.)
        log_inj = hk.get_parameter("log_inj", [], init=-0.3)
        log_dis = -3.
        alpha = hk.get_parameter("alpha", [], init=11 / 3)

        k_inj = 10 ** (-log_inj)
        k_dis = 10 ** (-log_dis)

        sigma = 10 ** log_sigma

        k_int = jnp.geomspace(k_inj / 20, k_dis * 20, 1000)
        norm = jsi.trapezoid(
            4 * jnp.pi * k_int ** 3 * jnp.exp(-(k_inj / k_int) ** 2) * jnp.exp(-(k_int / k_dis) ** 2) * (k_int) ** (
                -alpha), x=jnp.log(k_int))
        res = jnp.where(k > 0, jnp.exp(-(k_inj / k) ** 2) * jnp.exp(-(k / k_dis) ** 2) * k ** (-alpha), 0.)

        return sigma ** 2 * res / norm


p3d = hk.without_apply_rng(hk.transform(lambda k: KolmogorovPowerSpectrum()(k)))
scales = jnp.geomspace(0.1, 1., 200)  # In R500

joint_files = [
    f'results/joint_parameters/abs_0_1_joint_fit.posterior'
]

category = [
    r'X-COP 0-1 $R_{500}$'
]

# Planck

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

chains_k_inj_noise = chains_noise[:,:,0].flatten()
chains_slope_noise = chains_noise[:,:,1].flatten()
chains_log_norm_noise = chains_noise[:,:,2].flatten()
chains_norm_noise = 10**chains_log_norm_noise

#R = np.geomspace(cluster_info.R500_to_pix(0.1),cluster_info.R500_to_pix(1),500)
K = 1/cluster_info.R500_to_pix(scales)

pow_spec_chains_ell = chains_norm_ell[None,...]*np.exp(-chains_k_inj_ell[None,...]/K[...,None])*K[...,None]**(-chains_slope_ell[None,...])
med_pow_spec_ell = np.median(pow_spec_chains_ell,axis=1)
std_pow_spec_ell = np.std(pow_spec_chains_ell,axis=1)/np.sqrt(np.shape(pow_spec_chains_ell)[1])

pow_spec_chains_noise = chains_norm_noise[None,...]*np.exp(-chains_k_inj_noise[None,...]/K[...,None])*K[...,None]**(-chains_slope_noise[None,...])
med_pow_spec_noise = np.median(pow_spec_chains_noise,axis=1)
std_pow_spec_noise = np.std(pow_spec_chains_noise,axis=1)/np.sqrt(np.shape(pow_spec_chains_noise)[1])

font = {
    'size':20,
    'weight':'normal'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

#plt.plot(scales*20/1024,med_pow_spec_ell*K**3,'-b')
#plt.plot(scales*20/1024,med_pow_spec_noise*K**3,'-r')
#plt.fill_between(scales, *np.percentile(pow_spec_chains_noise,[16,84],axis=1)*K**3,color='orange',alpha=0.5)
plt.fill_between(scales, *np.percentile(pow_spec_chains_ell,[16,84],axis=1)*K**3,color='blue',alpha=0.5)

# Plot

plt.figure(1)

colors = ['red']
for file, cat, color in zip(joint_files, category, colors):
    power_spectrum = az.extract(az.from_netcdf(file))
    theta = jnp.asarray(power_spectrum.theta)
    turb_pars = {'kolmogorov_power_spectrum':
        {
            'log_sigma': theta[0, :],
            'log_inj': theta[1, :],
            'alpha': theta[2, :],
        }
    }

    p3d_sample = jax.vmap(lambda pars_tree: p3d.apply(pars_tree, 1 / scales))(turb_pars)
    plt.fill_between(scales, *np.percentile(p3d_sample, [16, 84], axis=0) * (1 / scales) ** 3 * gamma, alpha=0.5, color=color,
                     label=cat)
plt.legend([r'Planck (this work)',r'XMM (Dupourqué 2023)'],prop={'size':20},loc='lower left')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Echelle spatiale $(R_{500})$')
plt.ylabel(r'$P_{3D,\delta P}(k) \times k^3$')
plt.xlim([np.min(scales),np.max(scales)])
plt.ylim([5e-5,2e-3])
plt.tick_params(axis='x',which='minor',bottom=False)
plt.grid()
#plt.title('Spectre des fluctuations de pression pour les amas XCOP')
plt.show()