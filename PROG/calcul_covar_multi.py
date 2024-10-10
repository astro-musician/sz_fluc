import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from multiprocessing import Pool, cpu_count, set_start_method
import time
import pickle
from cluster_SZ import power_spectrum, healpix, pathfinder
from cluster_SZ import cluster_info, mean_model

"""
Calcul de la matrice de covariance des healpix pour un amas et une zone autour du centre spécifiques.
Cette matrice a pour but d'être utilisée dans le fit du modèle moyen (chi2).
La méthode est parallélisée.
"""

amas='A2319'
n_R500=3
method='transfert'

prog_path=pathfinder.prog_path()
tmpdir_path=pathfinder.tmpdir_path()

PSgrid=jnp.fft.fftshift(power_spectrum.PS_noise_amas(amas).PS_grid(10,savefile=False))
shape=jnp.shape(PSgrid)
im_size_x, im_size_y = jnp.shape(PSgrid)

image = cluster_info.image(amas,10)
mask_image = power_spectrum.mask_SZ(amas)
image_noise = np.delete(image.flatten(),mask_image.flatten()==0)
offset = np.mean(image_noise)

def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

def noise_simu():
	white_noise=random.normal(rng_key(),shape=shape)
	return cluster_info.reduce_n_R500(jnp.real(jnp.fft.ifft2(jnp.sqrt(PSgrid)*jnp.fft.fft2(white_noise))), n_R500)+offset

transfert=healpix.transfert(amas,n_R500,savefile=True).matrix
def noise_simu_healpix():
    return jnp.dot(transfert,noise_simu().flatten())

n_simu = 10000
n_processes = cpu_count()
def noise_samples(i):
    samples = []
    noisepath = tmpdir_path+'covar/noise_samples/' + amas + '_' + method + '_' + str(n_R500) + 'R500_noise_samples_' + str(i) + '.npy'
    for j in range(1,n_simu//n_processes+1):
        samples.append(noise_simu_healpix())
        if ((j+1)%100==0):
            print('CPU '+str(i+1)+' : '+str(j+1)+' simulations')
    with open(noisepath,'wb') as f:
        jnp.save(f,samples)
    return

if __name__=='__main__':

    set_start_method('forkserver')
    savepath=tmpdir_path+"covar/"+amas+"_covar_"+method+"_"+str(n_R500)+"R500.npy"

    print('START')
    time_start = time.time()

    with Pool(n_processes) as pool:
        pool.map(noise_samples,list(range(n_processes)))

    with open(tmpdir_path+'covar/noise_samples/'+amas+'_'+method+'_'+str(n_R500)+'R500_noise_samples_0.npy','rb') as f:
        noise_list=jnp.load(f)

    for i in range(1,n_processes):
        noisepath = tmpdir_path+'covar/noise_samples/' + amas + '_' + method + '_' + str(n_R500) + 'R500_noise_samples_' + str(i) + '.npy'
        with open(noisepath,'rb') as f:
            samples=jnp.load(f)
            noise_list = jnp.concatenate((noise_list,samples),axis=0)
    
    noise_list = noise_list.T
    cov_matrix = jnp.cov(noise_list)

    time_stop = time.time()

    print('Durée de calcul : ',round(time_stop-time_start),'s')

    with open(savepath,"wb") as file:
        jnp.save(file,cov_matrix)
	
