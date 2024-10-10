import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from cluster_SZ import cluster_info

AMAS = ['A85','A644','A1644','A1795','A2029','A2142','A2255','A2319','A3158','A3266','ZW1215']

ra = [cluster_info.skycoord(amas).l.wrap_at(180*u.degree).radian for amas in AMAS]
dec = [cluster_info.skycoord(amas).b.radian for amas in AMAS]

font = {
    'size':15,
    'weight':'semibold'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111,projection="mollweide")
ax.scatter(ra,dec)
for i, txt in enumerate(AMAS):
    ax.annotate(txt,(ra[i],dec[i]))
ax.grid(True)

plt.show()