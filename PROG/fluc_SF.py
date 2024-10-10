from cluster_SZ import SF, cluster_info, healpix, pathfinder
import pickle
import numpy as np
import matplotlib.pyplot as plt
tmpdir_path = pathfinder.tmpdir_path()

amas='A2319'
n_R500=3
n_R500_low=0
n_R500_high=2
im=cluster_info.fluc(amas,n_R500)
n_bins=20

font = {
    'size':15,
    'weight':'semibold'
}

transfert=healpix.transfert(amas,n_R500).matrix
healpix_vect=np.dot(transfert,im.flatten())
spnorm=cluster_info.spnorm(n_R500)
true_mask=(spnorm>cluster_info.R500_to_pix(n_R500_low))&(spnorm<cluster_info.R500_to_pix(n_R500_high))
healpix_spnorm=np.dot(transfert,spnorm.flatten())
healpix_mask=(healpix_spnorm>cluster_info.R500_to_pix(n_R500_low))&(healpix_spnorm<cluster_info.R500_to_pix(n_R500_high))

im_size = np.shape(im)[0]
bins = np.geomspace(2*cluster_info.healpix_mean_side(amas),cluster_info.R500_to_pix(n_R500_high-n_R500_low),n_bins+1)
print('bins : ',bins)
lows=bins[:-1]
highs=bins[1:]

X = cluster_info.pix_to_R500(lows+highs)/2
X_err = cluster_info.pix_to_R500(highs-lows)/2
Y = SF.structure_function(healpix_vect,amas,n_R500,lows,highs,healpix_mask)
print('SF : ',Y)
Y_err = SF.structure_function_err(healpix_vect,amas,n_R500,lows,highs,healpix_mask)

with open(tmpdir_path+'SF/'+amas+'_noise_SF.pkl','rb') as f:
    noise=pickle.load(f)

with open(tmpdir_path+'SF/'+amas+'_fluc_SF.npy','wb') as f:
    Y_mod = np.delete(Y,1)
    np.save(f,Y)

X_noise=noise['X']
Xerr_noise=noise['X_err']
Y_noise=noise['Y']
Yerr_noise=noise['Y_err']/2
#Ystd_noise=noise['Y_std']/2

font = {
    'size':20,
    'weight':'semibold'
}
plt.rcParams.update({'font.size':font['size'],'font.weight':font['weight']})

plt.figure(1)
plt.errorbar(X,Y,yerr=Y_err,fmt='. ',barsabove=True,capsize=10.0)
plt.errorbar(X_noise,Y_noise,yerr=Yerr_noise,fmt='. ',barsabove=True,capsize=10.0)
plt.legend(['Fluctuations','Bruit'])
plt.xscale('log')
plt.xlabel(r'$\mathbf{R_{500}}$',fontdict=font)
plt.yscale('log')
plt.ylabel(r'$\Delta Y^2$',fontdict=font)
plt.title('Structure fonction '+amas+' de '+str(n_R500_low)+' à '+str(n_R500_high)+r'$\mathbf{R_{500}}$',fontdict=font)
plt.grid()

#plt.figure(2)
#plt.imshow(im*true_mask)
#plt.colorbar()

plt.show()