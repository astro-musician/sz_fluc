"""
Routine pour rebin de pixels 1024*1024 en healpix
"""

import numpy as np
import jax.numpy as jnp
from astropy.io import fits
import os
from os.path import exists
from .pathfinder import prog_path, tmpdir_path
from .cluster_info import pix_image, pix_image_noise, reduce_n_R500, healpix_mean_side

prog_path=prog_path()
tmpdir_path='../'

	
#---------------------------------------------------
# MATRICE DE TRANSFERT PIXELS -> HEALPIX
#---------------------------------------------------

class form:

	def __init__(self,amas):
		self.amas=amas
		PIXEL=pix_image(amas,3)
		unique_PIXEL = np.unique(PIXEL)
		healpix_list = []
		for h in unique_PIXEL:
			h_form=np.where(PIXEL==h)
			if len(h_form[0])>15:
				healpix_list.append(h_form)
		self.list=healpix_list

class transfert:
	"""
	Matrice de transfert pixels 1024 -> healpix
	"""

	def __init__(self,amas,n_R500,savefile=False):
		self.amas=amas
		self.n_R500=n_R500
		PIXEL=pix_image(amas,n_R500).flatten()
		PIXEL_sq=pix_image(amas,n_R500)
		unique_PIXEL = np.unique(PIXEL)
		
		savepath=tmpdir_path+'transfert/'+amas+'_'+str(n_R500)+'R500.npy'
		
		if savefile:
			matrix=np.zeros((len(unique_PIXEL),len(PIXEL)))
			for h in range(len(unique_PIXEL)):
				mask = PIXEL == unique_PIXEL[h]
				matrix[h,:] = mask/np.sum(mask)
			
			self.matrix=jnp.array(matrix)
				
			with open(savepath,'wb') as f:
				jnp.save(f,matrix)
				
		else:
			if exists(savepath):
				with open(savepath,'rb') as f:
					self.matrix=jnp.load(f)
					
			else:
				matrix=np.zeros((len(unique_PIXEL),len(PIXEL)))
				for h in range(len(unique_PIXEL)):
					mask = PIXEL == unique_PIXEL[h]
					matrix[h,:] = mask/np.sum(mask)
				self.matrix=jnp.array(matrix)
		
		

class healpix_index:
	"""
	Liste des indices des pixels au centre de chaque healpix (rebin simplifié)
	"""

	def __init__(self,amas,n_R500,savefile=False):
		self.amas=amas
		self.n_R500=n_R500
		PIXEL=pix_image(amas, n_R500)
		PIXEL_flat=pix_image(amas,n_R500).flatten()
		unique_PIXEL, index_PIXEL=np.unique(PIXEL_flat,return_index=True)
		indices=np.zeros((len(unique_PIXEL),2))

		for h in range(len(unique_PIXEL)):
			x_healpix, y_healpix = np.where(PIXEL==unique_PIXEL[h])
			ind_x, ind_y = np.mean(x_healpix), np.mean(y_healpix)
			indices[h,:]=np.array([ind_x,ind_y])

		self.indices=indices

		savepath=tmpdir_path+'transfert/'+amas+'_'+str(n_R500)+'R500_healpix_index.npy'

		if savefile:
			with open(savepath,'wb') as f:
				jnp.save(f,indices)

class healpix_index_noise:
    
    def __init__(self,amas,xc,yc,n_R500):
        self.amas=amas
        self.n_R500=n_R500
        self.xc=xc
        self.yc=yc
        PIXEL = pix_image_noise(amas,xc,yc,n_R500)
        PIXEL_flat = PIXEL.flatten()
        unique_PIXEL, index_PIXEL=np.unique(PIXEL_flat,return_index=True)
        indices=np.zeros((len(unique_PIXEL),2))
        
        for h in range(len(unique_PIXEL)):
            x_healpix, y_healpix = np.where(PIXEL==unique_PIXEL[h])
            ind_x, ind_y = np.mean(x_healpix), np.mean(y_healpix)
            indices[h,:]=np.array([ind_x,ind_y])
            
        self.indices=indices

class transfert_rebin:
	"""
	Méthode de rebin d'image avec matrice de transfert. Cette dernière est rapidement lourde et longue à calculer (carrée de côté n_pixels**2).
	"""

	def __init__(self,amas,n_R500,savefile=False):
		self.amas=amas
		self.n_R500=n_R500
		PIXEL=pix_image(amas,n_R500).flatten()
		unique_PIXEL = np.unique(PIXEL)
		savepath = tmpdir_path+'transfert/' + amas + '_' + str(n_R500) + 'R500_rebin.npy'

		if savefile:
			matrix = np.zeros((len(PIXEL), len(PIXEL)))
			for h in range(len(unique_PIXEL)):
				mask = PIXEL == unique_PIXEL[h]
				matrix[mask, :] = mask / np.sum(mask)

			self.matrix = jnp.array(matrix)

			with open(savepath, 'wb') as f:
				jnp.save(f, matrix)

		else:
			if exists(savepath):
				with open(savepath, 'rb') as f:
					self.matrix = jnp.load(f)

			else:
				matrix = np.zeros((len(PIXEL), len(PIXEL)))
				for h in range(len(unique_PIXEL)):
					mask = PIXEL == unique_PIXEL[h]
					matrix[mask, :] = mask / np.sum(mask)
				self.matrix = jnp.array(matrix)