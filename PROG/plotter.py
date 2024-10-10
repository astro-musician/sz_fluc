import torch
import numpy as np
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import jax.random as random
import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from cluster_SZ import cluster_info, power_spectrum, mock_fluc, mean_model, pathfinder

amas='A2029'
n_R500=10

image=np.array(cluster_info.image(amas,n_R500))
mask=power_spectrum.mask_SZ(amas)
X,Y=np.meshgrid(np.arange(cluster_info.pix_shape(n_R500)[0]),np.arange(cluster_info.pix_shape(n_R500)[1]))
fwhm_pix = 7*1024/20/cluster_info.theta500(amas)
print(fwhm_pix)

x_beam, y_beam = 50, 50
beam_loc = ((X-x_beam)**2+(Y-y_beam)**2)<fwhm_pix**2
circle = plt.Circle((x_beam,y_beam),fwhm_pix,color='orange')

font = {
    'weight':'normal',
    'size':12,
}

plt.imshow(image*mask)
plt.xlabel(r'1024 pix = 20 $R_{500}$',fontdict=font)
plt.ylabel(r'1024 pix = 20 $R_{500}$',fontdict=font)
fig = plt.gcf()
ax = fig.gca()
ax.add_patch(circle)
plt.text(x_beam,y_beam+fwhm_pix*3,'Beam',backgroundcolor='w',fontweight='bold',size=10)
plt.colorbar(label='Compton parameter')
plt.show()