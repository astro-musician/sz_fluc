import torch

from sbi import utils as utils
from sbi.inference import SNLE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import jax.numpy as jnp
import numpy as np
import pickle
from cluster_SZ import mean_model, cluster_info, power_spectrum

amas='A2319'
n_R500=3
n_R500_high = 2
n_R500_low = 0
n_loops = 19

double_simus = False

spnorm = cluster_info.spnorm(n_R500)
mask = (spnorm>cluster_info.R500_to_pix(n_R500_low))&(spnorm<cluster_info.R500_to_pix(n_R500_high))

low_inj = float(1/cluster_info.R500_to_pix(5))
high_inj = float(1/cluster_info.R500_to_pix(0.1))
low_slope = 10/3
high_slope = 4
low_log_norm = -10
high_log_norm = -jnp.log(2.4)

prior=utils.BoxUniform(low=torch.Tensor([low_inj,low_slope,low_log_norm]),high=torch.Tensor([high_inj,high_slope,high_log_norm]))
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Map simus
x_path = 'posteriors/simulations/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu.npy'
x_path_2 = 'posteriors/simulations/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu_2.npy'
with open(x_path,'rb') as f:
    x=np.load(f)

print(x[0])

if double_simus:
    n_loops = 38
    with open(x_path_2,'rb') as f:
        x2=np.load(f)

    print(np.shape(x))
    print(np.shape(x2))
    x = np.concatenate((x,x2[:-5005]),axis=0)

print(np.shape(x))
x_obs=torch.from_numpy(np.array(x))

# PS simus
#mock_ps_path = 'posteriors/simulations/'+amas+'_'+str(n_R500_mask_low)+'_to_'+str(n_R500_mask_high)+'R500_mock_power_spectra.pkl'
#with open(mock_ps_path,'rb') as f:
#    x_PS=pickle.load(f)

# theta simus

with open('posteriors/simulations/'+amas+'_fluc_map_simu_theta_0.npy','rb') as f:
    theta=np.load(f)

for n in range(1,n_loops):
    with open('posteriors/simulations/'+amas+'_fluc_map_simu_theta_'+str(n)+'.npy','rb') as f:
        new_theta = np.load(f)
    theta = np.concatenate((theta,new_theta),axis=0)

print(np.shape(theta))
theta = torch.from_numpy(np.array(theta))

#restriction_estimator=utils.RestrictionEstimator(prior=prior)
#restriction_estimator.append_simulations(theta, x)
#classifier=restriction_estimator.train()
#restricted_prior = restriction_estimator.restrict_prior()
#restricted_theta, restricted_x, _ = restriction_estimator.get_simulations()

#restricted_prior, num_parameters, restricted_prior_returns_numpy = process_prior(restricted_prior)

inference=SNLE(prior=prior)
inference = inference.append_simulations(theta, x_obs) # Entrée des simulations
density_estimator=inference.train() # Entraînement du réseau de neurones avec les simulations ci-dessus
posterior=inference.build_posterior(density_estimator) # Fonction traçant la posterior à partir d'observations.
# Il est crucial de pouvoir sauver puis charger cet objet. Ça se fait avec pickle.

with open('posteriors/'+amas+'_fluc_likelihood_'+str(n_R500_low)+'_to_'+str(n_R500_high)+'R500.pkl','wb') as f:
    pickle.dump(posterior,f)