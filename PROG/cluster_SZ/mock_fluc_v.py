import numpy as np
import jax.numpy as jnp
import jax.random as random
# import jax.scipy.integrate as jsi

def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

def mock_PS(k,kcut,slope,norm):
    return norm[...,None,None,None]*jnp.exp(-kcut[...,None,None,None]/jnp.clip(k,a_min=1e-3,a_max=jnp.inf)[None,...])/jnp.clip(k,a_min=1e-3,a_max=jnp.inf)[None,...]**slope[...,None,None,None]

# Cette fonction devra certainement être fournie lors de l'appel pour que le code puisse tourner sous jit (grille knorm)
def mock_PS_cube(knorm_3D,kcut,slope,norm):
    return mock_PS(knorm_3D,kcut,slope,norm)

# Idem
def mock_fluc_cube(n_sim,cube_size,L_size,knorm_3D,kcut,slope,norm):
    white_noise=random.normal(rng_key(),shape=(n_sim,cube_size,cube_size,L_size))
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(white_noise,axes=(1,2,3))*jnp.sqrt(mock_PS_cube(knorm_3D,kcut,slope,norm)),axes=(1,2,3)))
