import numpy as np
import jax.numpy as jnp
import jax.random as random
from jax import jit
import jax.scipy.integrate as jsi
from time import time
from cluster_SZ import healpix, power_spectrum, cluster_info, mock_fluc
import matplotlib.pyplot as plt
from astropy.io import fits

n_R500=2
amas='A2142'
spnorm=cluster_info.spnorm(n_R500)
cube_size=cluster_info.pix_shape(n_R500)[0]
knorm_3D=cluster_info.knorm_3D(n_R500)

#transfert_rebin=healpix.transfert_rebin(amas,n_R500,savefile=False).matrix

def rng_key():
	return random.PRNGKey(np.random.randint(0,1e6))

def mock_PS(k,kcut,slope,norm):
    return norm*jnp.exp(-kcut/jnp.clip(k,a_min=1e-3,a_max=jnp.inf))/jnp.clip(k,a_min=1e-3,a_max=jnp.inf)**slope

# Cette fonction devra certainement être fournie lors de l'appel pour que le code puisse tourner sous jit (grille knorm)
def mock_PS_cube(kcut,slope,norm):
    return mock_PS(knorm_3D,kcut,slope,norm)

# Idem
def mock_fluc_cube(kcut,slope,norm):
    white_noise=random.normal(rng_key(),shape=(cube_size,cube_size,cube_size))
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(white_noise,axes=(0,1,2))*jnp.sqrt(mock_PS_cube(kcut,slope,norm)),axes=(0,1,2)))

# Rajouter le modèle moyen avec des paramètres pris dans les samples

def mock_fluc_image(kcut,slope,norm):
    return jsi.trapezoid(mock_fluc_cube(kcut,slope,norm),axis=2,dx=1)

mask=jnp.ones((cube_size,cube_size))
unmasked_zone= mask==1
K_amp = jnp.geomspace(2/(jnp.sqrt(2)*cube_size),jnp.sqrt(2)/2,30)
def PS_mock_fluc_image(kcut,slope,norm,K_amp,spnorm):
    return mock_fluc.power_spectrum(mock_fluc_image(kcut,slope,norm),mask,K_amp,unmasked_zone,spnorm)

kcut=1/cluster_info.R500_to_pix(0.5)
slope=2
norm=1e-9

start_nojit=time()
PS_nojit=PS_mock_fluc_image(kcut,slope,norm,K_amp,spnorm)
end_nojit=time()

start_jit=time()
PS_jit=jit(PS_mock_fluc_image)(kcut,slope,norm,K_amp,spnorm)
end_jit=time()

print('No jit',round(end_nojit-start_nojit,2),'s')
print('Jit',round(end_jit-start_jit,2),'s')

plt.plot(1024/20*K_amp,PS_jit,'. ')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{500}^{-1}$')
plt.ylabel('Mock PS')
plt.grid()
plt.show()