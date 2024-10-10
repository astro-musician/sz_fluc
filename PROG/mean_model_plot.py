import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle
from cluster_SZ import mean_model, pathfinder, cluster_info

amas='A2029'
n_R500=3

font = {
    'size':20,
    'weight':'semibold'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

image = cluster_info.image(amas,n_R500)
mm_bfit = cluster_info.mm_bfit(amas,n_R500)

fig, axes = plt.subplots(ncols=2,nrows=1)
vmin=np.min(image)
vmax=np.max(image)

im=axes.flat[0].imshow(mm_bfit,vmin=vmin,vmax=vmax,interpolation=None)
axes.flat[0].title.set_text('Modèle moyen')
axes.flat[1].imshow(image,vmin=vmin,vmax=vmax,interpolation=None)
axes.flat[1].title.set_text('Image')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.title(amas)

plt.show()