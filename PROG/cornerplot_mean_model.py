import numpy as np
import pickle
from chainconsumer import Chain, ChainConsumer, ChainConfig
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from cluster_SZ import pathfinder
tmpdir_path = pathfinder.tmpdir_path()

amas='ZW1215'

filepath1 = tmpdir_path+'mean_models/'+amas+'_transfert_MCMC_chains.pkl'

c=ChainConsumer()

with open(filepath1, 'rb') as f:
    chains1 = pickle.load(f)

n_walkers, n_iter = np.shape(chains1['angle'])

angle1 = chains1['angle'].flatten()
ell1 = chains1['ell'].flatten()
xc1 = chains1['xc'].flatten()
yc1 = chains1['yc'].flatten()
P01 = chains1['P0'].flatten()
c5001 = chains1['c_500'].flatten()
beta1 = chains1['beta'].flatten()

d1 = {r'angle':angle1,r'ell':ell1,r'$x_c$':xc1,r'$y_c$':yc1,r'$P_0$':P01,r'$c_{500}$':c5001,'beta':beta1}
df1 = pd.DataFrame(data=d1)

c.add_chain(Chain(samples=df1,name=amas,show_contour_labels=True,marker_style='*'))
#c.set_override(ChainConfig(sigmas=[0,1,2]))
fig=c.plotter.plot()
plt.show()

