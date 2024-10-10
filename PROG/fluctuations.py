import numpy as np
import pickle
import matplotlib.pyplot as plt
from cluster_SZ import mean_model, power_spectrum, cluster_info

amas='A3266'
n_R500=2
n_R500_mask_high = 2
n_R500_mask_low = 0
spnorm = cluster_info.spnorm(n_R500)
mask = (spnorm>cluster_info.R500_to_pix(n_R500_mask_low))&(spnorm<cluster_info.R500_to_pix(n_R500_mask_high))

spnorm = cluster_info.spnorm(n_R500)
knorm = cluster_info.knorm(n_R500)

mean_map = cluster_info.mm_bfit(amas,n_R500)
fluc_map = cluster_info.fluc(amas,n_R500)

plt.figure(1)
plt.imshow(fluc_map,interpolation=None)
plt.title('fluctuations '+amas+' à '+str(n_R500)+r'$R_{500}$')
plt.colorbar()

plt.figure(2)
fluc_PS = power_spectrum.PS_fluc_amas(amas,n_R500,mask=mask,k_min=1/cluster_info.R500_to_pix(2),k_max=1/cluster_info.R500_to_pix(0.05)).PS(savefile=True)
noise_PS = power_spectrum.PS_noise_amas(amas).PS(savefile=True,k_min=1/cluster_info.R500_to_pix(2),k_max=1/cluster_info.R500_to_pix(0.05))
plt.plot(power_spectrum.R500_to_pix(fluc_PS[0]),(20/1024)**2*fluc_PS[1],' .',power_spectrum.R500_to_pix(noise_PS[0]),(20/1024)**2*noise_PS[1],'. ')
plt.legend(['fluc','noise'])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{500}^{-1}$')
plt.ylabel('PS')
plt.title(amas+' fluc power spectrum')
plt.grid()

plt.show()