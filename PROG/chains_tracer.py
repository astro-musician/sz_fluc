#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:40:59 2024

@author: stagiaire
"""

import pickle
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

amas='A2319'
rebin_method='transfert' # Cette ligne devrait rester la même dans tous les cas (elle utilise la matrice de transfert pour créer un vecteur de pixels)
filepath = 'mean_models/'+amas+'_'+rebin_method+'_MCMC_chains.pkl'

with open(filepath,'rb') as f:
    chains = pickle.load(f)

print(np.median(chains['angle'],axis=1))
new_chains = {
    'xc':chains['xc'][:3],
    'yc':chains['yc'][:3],
    'angle':chains['angle'][:3],
    'ell':chains['ell'][:3],
    'P0':chains['P0'][:3],
    'c_500':chains['c_500'][:3],
    'beta':chains['beta'][:3]
}

#with open(filepath,'wb') as f:
#    pickle.dump(new_chains,f)

#with open(filepath+'_old','wb') as f:
#    pickle.dump(chains,f)

dataset = az.convert_to_inference_data(new_chains)

with az.style.context('arviz-darkgrid', after_reset=True):
    az.plot_trace(dataset, compact=True)

print('----------------------')
print('R HAT')
print(az.rhat(dataset))
print('----------------------')

plt.show()