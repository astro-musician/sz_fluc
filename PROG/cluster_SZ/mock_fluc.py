import numpy as np
import jax.numpy as jnp
import jax.random as random
#import jax.scipy.integrate as jsi

def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

def mock_PS(k,kcut,slope,norm):
    return norm*jnp.exp(-kcut/jnp.clip(k,a_min=1e-3,a_max=jnp.inf))/jnp.clip(k,a_min=1e-3,a_max=jnp.inf)**slope

# Cette fonction devra certainement être fournie lors de l'appel pour que le code puisse tourner sous jit (grille knorm)
def mock_PS_cube(knorm_3D,kcut,slope,norm):
    return mock_PS(knorm_3D,kcut,slope,norm)

# Idem
def mock_fluc_cube(cube_size,L_size,knorm_3D,kcut,slope,norm):
    white_noise=random.normal(rng_key(),shape=(cube_size,cube_size,L_size))
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(white_noise,axes=(0,1,2))*jnp.sqrt(mock_PS_cube(knorm_3D,kcut,slope,norm)),axes=(0,1,2)))

# Rajouter le modèle moyen avec des paramètres pris dans les samples

def mock_fluc_image(cube_size,L_size,knorm_3D,kcut,slope,norm):
    L = np.linspace(-5, 5, L_size)
    return jnp.trapz(mock_fluc_cube(cube_size,L_size,knorm_3D,kcut,slope,norm),axis=2,x=L)
