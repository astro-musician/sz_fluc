import torch
import numpy as np
import jax.numpy as jnp
from jax import jit
from time import time
from multiprocessing import Pool, set_start_method, cpu_count
from cluster_SZ import cluster_info, SF, healpix

amas='A2142'
n_R500=2.5
n_R500_low=0
n_R500_high=2
n_bins=20

n_loops = 100

with open('posteriors/simulations/'+amas+'_fluc_map_simu_x_healpix.npy','rb') as f:
    healpix_vects=jnp.load(f)

spnorm=cluster_info.spnorm(n_R500)
transfert=healpix.transfert(amas,n_R500).matrix
healpix_spnorm=np.dot(transfert,spnorm.flatten())
healpix_mask=(healpix_spnorm>cluster_info.R500_to_pix(n_R500_low))&(healpix_spnorm<cluster_info.R500_to_pix(n_R500_high))

bins = np.geomspace(2*cluster_info.healpix_mean_side(amas),cluster_info.R500_to_pix(n_R500_high-n_R500_low),n_bins+1)
lows=bins[:-1]
highs=bins[1:]
bin_matrix=SF.bin_matrix(amas,n_R500,lows,highs)

sfjit=jit(SF.structure_function_multi)
n_processes=cpu_count()
sf_per_cpu=int(len(healpix_vects)/n_processes)

def sf_multiprocess(X):
    hvects, i = X
    sfs = sfjit(hvects[i * sf_per_cpu:(i + 1) * sf_per_cpu], bin_matrix, healpix_mask)
    savepath = 'posteriors/simulations/temp/' + amas + '_temp_sf_' + str(i) + '.npy'
    with open(savepath, 'wb') as f:
        jnp.save(f, sfs)
    return

if __name__ == '__main__':
    set_start_method('forkserver')

    for n in range(n_loops):

        print('loop '+str(n+1))
        part_healpix_vects = healpix_vects[int(n*len(healpix_vects)/n_loops):int((n+1)*len(healpix_vects)/n_loops)]
        args = [[part_healpix_vects,j] for j in range(n_processes)]

        with Pool(n_processes)as pool:
            pool.map(sf_multiprocess,args)

        with open('posteriors/simulations/temp/'+amas+'_temp_sf_0.npy','rb') as f:
            SFs=jnp.load(f)

        for i in range(1,n_processes):
            with open('posteriors/simulations/temp/'+amas+'_temp_sf_'+str(i)+'.npy','rb') as f:
                sfs=jnp.load(f)
                SFs=jnp.concatenate((SFs,sfs),axis=1)

        print('Output shape : ',jnp.shape(SFs))
        with open('posteriors/simulations/temp/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu_'+str(n)+'.npy', 'wb') as f:
            jnp.save(f, SFs)

    with open('posteriors/simulations/temp/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu_0.npy','rb') as f:
        full_SFs=jnp.load(f)

    for n in range(1,n_loops):
        with open('posteriors/simulations/temp/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu_'+str(n)+'.npy','rb') as f:
            new_sfs=jnp.load(f)
            full_SFs=jnp.concatenate((full_SFs,new_sfs),axis=1)

    with open('posteriors/simulations/' + amas + '_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500_SF_simu.npy','wb') as f:
        jnp.save(f,full_SFs.T)
