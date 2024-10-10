import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from cluster_SZ import power_spectrum, pathfinder, healpix, cluster_info

amas='A2142'
n_R500=3

image = cluster_info.image(amas,10)
mask = power_spectrum.PS_noise_amas(amas).mask
transfert=healpix.transfert(amas,n_R500,savefile=False).matrix
image_noise = np.delete(image.flatten()*mask.flatten(),np.where(mask.flatten()==0))
k, PS = power_spectrum.PS_noise_amas(amas).PS(savefile=False)
PSgrid = power_spectrum.PS_noise_amas(amas).PS_grid(n_R500,savefile=False)

white_noise = np.random.normal(size=np.shape(PSgrid))
noise_simulation = np.real(np.fft.ifft2(np.fft.fft2(white_noise)*np.sqrt(PSgrid)))
image_rebin=np.zeros(np.shape(noise_simulation))
PIXEL=cluster_info.pix_image(amas,n_R500)
for h in np.unique(PIXEL):
    mask = PIXEL == h
    pixel_mean = np.mean(noise_simulation[mask])
    image_rebin[mask] = pixel_mean

noise_simulation_healpix = np.dot(transfert,noise_simulation.flatten())

print(np.std(noise_simulation_healpix)/np.std(image_noise))

font = {
    'size':20,
    'weight':'semibold'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

plt.figure(1)
plt.plot(power_spectrum.R500_to_pix(k),PS*(20/1024)**2,' .')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{500}^{-1}$')
plt.ylabel('PS')
plt.title(amas+' noise power spectrum')
plt.grid()

plt.figure(2)
plt.imshow(image_rebin,interpolation=None)
plt.colorbar()

counts_im, bins_im = np.histogram(image_noise,bins=20,range=(-1e-5,1e-5))
counts_sim, bins_sim = np.histogram(noise_simulation_healpix,bins=20,range=(-1e-5,1e-5))
fig, axes = plt.subplots(nrows=1,ncols=2)
fig.tight_layout()
plt.subplot(1,2,1)
plt.hist(bins_im[:-1],bins_im,weights=counts_im)
plt.xlim([-1e-5,1e-5])
plt.xlabel(r'$Y$')
#plt.ylim([0,1.1*np.max(np.array([counts_im,counts_sim]))])
plt.title('Image '+amas)
plt.subplot(1,2,2)
plt.hist(bins_sim[:-1],bins_sim,weights=counts_sim)
plt.xlim([-1e-5,1e-5])
#plt.ylim([0,1.1*np.max(np.array([counts_im,counts_sim]))])
plt.title('Simulation '+amas)

plt.show()