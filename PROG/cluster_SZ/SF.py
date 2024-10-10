from .healpix import healpix_index, healpix_index_noise, transfert
from .cluster_info import center_coord
from jax.lax import dot
import jax.numpy as jnp
from scipy.spatial import distance_matrix

def dist_matrix(amas,n_R500):
    H=healpix_index(amas,n_R500).indices
    return distance_matrix(H,H)

def dist_matrix_noise(amas,xc,yc,n_R500):
    H = healpix_index_noise(amas,xc,yc,n_R500).indices
    return distance_matrix(H,H)

def dist_to_center(amas,n_R500):
    H = healpix_index(amas, n_R500).indices
    center = center_coord(amas,n_R500)
    return jnp.sqrt((H[:,0]-center[0])**2+(H[:,1]-center[1])**2)

def bin_matrix(amas,n_R500,lows,highs):
    """
    (n_healpix,n_healpix) matrix
    =1 if healpix_i and healpix_j are within a distance >=low and <high to each other (lows and highs can be arrays)
    =0 otherwise
    """
    DM = dist_matrix(amas,n_R500)
    return (DM[None,...]>=lows[...,None,None])&(DM[None,...]<highs[...,None,None])

def bin_matrix_noise(amas,xc,yc,n_R500,lows,highs):
    """
    (n_healpix,n_healpix) matrix
    =1 if healpix_i and healpix_j are within a distance >=low and <high to each other (lows and highs can be arrays)
    =0 otherwise
    """
    DM = dist_matrix_noise(amas,xc,yc,n_R500)
    return (DM[None,...]>=lows[...,None,None])&(DM[None,...]<highs[...,None,None])

def diff_matrix(healpix_vect):
    """
    Returns the matrix of differences between every healpix values in image.
    image must already be under the form of a healpix vector (use transfert from healpix module)
    """
    return healpix_vect[:,None]-healpix_vect[None,:]

def diff_matrix_multi(healpix_vects):
    healpix_vects=healpix_vects.T
    return healpix_vects[None,:,:]-healpix_vects[:,None,:]

def structure_function(healpix_vect,amas,n_R500,lows,highs,healpix_mask):
    healpix_mask=jnp.int32(healpix_mask)
    mask = jnp.int32(bin_matrix(amas,n_R500,lows,highs))
    mask = mask*healpix_mask[None,...,None]*healpix_mask[None,None,...]
    return jnp.mean(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1),where=mask==1)
    #return jnp.sum(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1))/jnp.sum(mask,axis=(-2,-1))

def structure_function_err(healpix_vect,amas,n_R500,lows,highs,healpix_mask):
    mask = bin_matrix(amas,n_R500,lows,highs)
    bins_sizes = jnp.sum(mask,axis=(-2,-1))
    mask = mask*healpix_mask[None,...,None]*healpix_mask[None,None,...]
    return jnp.std(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1),where=mask==1)/jnp.sqrt(bins_sizes)

def structure_function_std(healpix_vect,amas,n_R500,lows,highs,healpix_mask):
    mask = bin_matrix(amas,n_R500,lows,highs)
    mask = mask*healpix_mask[None,...,None]*healpix_mask[None,None,...]
    return jnp.std(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1),where=mask==1)

def structure_function_noise(healpix_vect,amas,xc,yc,n_R500,lows,highs,healpix_mask):
    healpix_mask=jnp.int32(healpix_mask)
    mask = jnp.int32(bin_matrix_noise(amas,xc,yc,n_R500,lows,highs))
    mask = mask*healpix_mask[None,...,None]*healpix_mask[None,None,...]
    return jnp.mean(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1),where=mask==1)

def structure_function_noise_err(healpix_vect,amas,xc,yc,n_R500,lows,highs,healpix_mask):
    mask = bin_matrix_noise(amas,xc,yc,n_R500,lows,highs)
    bins_sizes = jnp.sum(mask,axis=(-2,-1))
    mask = mask*healpix_mask[None,...,None]*healpix_mask[None,None,...]
    return jnp.std(mask*diff_matrix(healpix_vect)[None,...]**2,axis=(-2,-1),where=mask==1)/jnp.sqrt(bins_sizes)
    

def structure_function_multi(healpix_vects,bin_matrix,healpix_mask):
    """
    Structure functions of several healpix vectors.
    Computation is made on a 4D array :
    dim 0 : bins
    dim 1-2 : distance matrices
    dim 3 : healpix vectors
    """
    mask = bin_matrix
    mask = mask * healpix_mask[None,..., None] * healpix_mask[None,None, ...]
    return jnp.sum(mask[...,None]*diff_matrix_multi(healpix_vects)[None,...]**2,axis=(1,2))/jnp.sum(mask,axis=(1,2))[...,None]

