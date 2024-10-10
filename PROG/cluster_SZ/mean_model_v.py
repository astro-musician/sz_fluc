#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:07:12 2024

@author: stagiaire

IMPORTANT : Le dossier data_planck doit être placé dans la même arborescence que le dossier contenant les programmes
qui utilisent ce package.

"""

from math import *
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial
# import jax.scipy.integrate as jsi
from os.path import exists
from .cluster_info import pix_to_R500, R500_to_pix, pix_image
from .pathfinder import prog_path

prog_path=prog_path()
    
prog=True
if prog:


#---------------------------------------------------
# TRANSFORMATIONS GÉOMÉTRIQUES 
#---------------------------------------------------

#Matrice de rotation dans le plan (x,y), ie le plan du ciel :

        def rotation_2d(angle):
            matrix = jnp.array([[jnp.cos(angle),jnp.sin(angle)],
                             [-jnp.sin(angle),jnp.cos(angle)]])
            # Permutation circulaire pour assurer aue les indices des simulations sont sur l'axe 0
            matrix = jnp.transpose(matrix, (2, 0, 1))
            return matrix

        def rotation_3d(angle):
            matrix = jnp.array([[jnp.cos(angle),-jnp.sin(angle),jnp.zeros(jnp.shape(angle))],
                                [jnp.sin(angle),jnp.cos(angle),jnp.zeros(jnp.shape(angle))],
                                [jnp.zeros(jnp.shape(angle)),jnp.zeros(jnp.shape(angle)),jnp.ones(jnp.shape(angle))]])

            matrix = jnp.transpose(matrix,(2,0,1))
            return matrix

# Matrice de dilatation dans le plan (x,y) :
    
        def dilat(Dx,Dy):
            """
            Dx : Dilatation le long de x
           Dy : Dilatation le long de y
    
            """
            matrix = jnp.array([[Dx, jnp.zeros(jnp.shape(Dx))],
                             [jnp.zeros(jnp.shape(Dy)), Dy]])
            # Permutation circulaire pour assurer que les indices des simulations sont sur l'axe 0
            matrix = jnp.transpose(matrix, (2, 0, 1))
            return matrix

        def dilat_3d(Dx,Dy):
            matrix = jnp.array([[Dx,jnp.zeros(jnp.shape(Dx)),jnp.zeros(jnp.shape(Dx))],
                                [jnp.zeros(jnp.shape(Dx)),Dy,jnp.zeros(jnp.shape(Dx))],
                                [jnp.zeros(jnp.shape(Dx)),jnp.zeros(jnp.shape(Dx)),jnp.ones(jnp.shape(Dx))]])

            # Permutation circulaire pour assurer que les indices des simulations sont sur l'axe 0
            matrix = jnp.transpose(matrix,(2,0,1))
            return matrix

# Matrice de transformation elliptique d'axes (x,y) :

        def dilat_ell(ell):
            """
            e : ellipticité
            """
            return dilat(1/jnp.sqrt(1-jnp.clip(ell,0,0.99)**2),jnp.sqrt(1-jnp.clip(ell,0,0.99)**2))

        def dilat_ell_3d(ell):
            return dilat_3d(1/jnp.sqrt(1-jnp.clip(ell,0,0.99)**2),jnp.sqrt(1-jnp.clip(ell,0,0.99)**2))

#--------------------------------------------------- 
# MODÉLISATION DU PROFIL DE PRESSION ELLIPTIQUE
#---------------------------------------------------

# Profil de pression sphérique de Ghirardini :
    
        def pression_ghirardini(P500,r,P0,c_500,beta):
            """
            Profil de pression sphérique dérivé de l'article de Ghirardini
            r : distance au centre en R500
            P500 : échelle de pression
            """
            # P0 = 5.68
            # c_500= 1.49
            gamma = 0
            alpha = 1.33
            # beta = 4.4
            r=jnp.clip(r,a_min=0.01, a_max=jnp.inf) # écrète r
            return P500*P0[...,None,None,None]/( (c_500[...,None,None,None]*r)**gamma * ( 1 + (c_500[...,None,None,None]*r)**alpha )**((beta[...,None,None,None] - gamma)/alpha))

# A PARTIR D'ICI, TOUT DOIT ÊTRE DÉFINI DE MANIÈRE À POUVOIR ÊTRE APPELÉ
# EN FONCTION DES PARAMÈTRES À FIT :
# ell (ellipticité), angle (rotation), P0, c500, beta

# On donne d'abord les grandeurs en SI
        

# Coefficient de proportionnalité entre le paramètre Compton et la pression intégrée
# en échelle de longueur R500 (pas besoin de rescale la longueur, voir notes) :


        def pression(P500,x,y,l,P0,c_500,beta):
            """
            Pression électronique en fonction des coordonnées sphériques.
            Il faut d'abord faire un changement de coordonnées x,y vers le système elliptique si on veut 
            utiliser la pression dans le modèle.
            l : coordonnée sur l'axe de la ligne de visée
            """
    # Jusqu'ici on était en pixels, donc on passe en R500. L l'est déjà.
            x_R500, y_R500, l_R500 = pix_to_R500(x), pix_to_R500(y), l
            r = jnp.sqrt(x_R500**2+y_R500**2+l_R500**2)
            return pression_ghirardini(P500,r,P0,c_500,beta)

        L = jnp.geomspace(R500_to_pix(0.01),R500_to_pix(5),100) # linspace logarithmique
        L = jnp.geomspace(0.01,5,100)
        def pression_integree(P500,x,y,P0,c_500,beta):
            """
            Intègre le profil de pression le long de la ligne de visée aux coordonnées (x,y) du plan du ciel.
            Les unités de longueur sont en R500.
            """
    
            P_int = 2*jnp.trapz(pression(P500,x[..., None],y[..., None], L[None, None, :],P0,c_500,beta), axis=-1, x=L) # trapeze
            return P_int

        def transformation_elliptique(x,y,ell,angle,xc,yc):
            """
            x,y : position dans le plan (en pixels)
           ell : ellipticité
           angle : angle de rotation dans le plan du ciel
           xc, yc : origine du nouveau repère
           """
    # On commence par redéfinir l'origine
            x = x-jnp.array([xc])[...,None]
            y = y-jnp.array([yc])[None,...]
    # On applique la dilatation PUIS la rotation (sinon on dilatera toujours en vertical/horizontal)
            matrice_transfo = jnp.dot(dilat_ell(ell),rotation_2d(angle))
            #X_ell =  jnp.matmul(matrice_transfo[..., None, None].T , X_center[None, ...].T )
            return jnp.einsum('ji, imn -> jmn',matrice_transfo,jnp.array([x,y])) #X_ell[..., 0].T

        def transformation_elliptique_3D(x,y,ell,angle,xc,yc):
            """
            x,y : position dans le plan (en pixels)
           ell : ellipticité
           angle : angle de rotation dans le plan du ciel
           xc, yc : origine du nouveau repère
           """
    # On commence par redéfinir l'origine
            length = len(xc)
            xc = xc.reshape(length, 1)
            yc = yc.reshape(length, 1)
            x = x[None,...] - xc[:, :, None, None]
            y = y[None,...] - yc[:, None, :, None]
            # print(jnp.shape(x),jnp.shape(y),flush=True)
    # On applique la dilatation PUIS la rotation (sinon on dilatera toujours en vertical/horizontal)
            matrice_transfo = jnp.matmul(dilat_ell(ell),rotation_2d(angle))
            return jnp.einsum('kji, ikmnp -> jkmnp',matrice_transfo,jnp.array([x,y]))


#---------------------------------------------------
# MODÉLISATION DE LA CARTE DE PARAMÈTRE COMPTON
#---------------------------------------------------

# Intégration le long de la ligne de visée en (x,y) (ATTENTION A L'ÉCHELLE) :
        
        thomson_cs = 6.65e-29
        me = 9.1e-31
        c=3e8
        coeff_Y_pint = thomson_cs/(me*c**2)
    
        def Y(P500,x,y,xc,yc,angle,ell,P0,c_500,beta,transfo_grid):
            """
            Renvoie le paramètre Compton en (x,y) suite à la transformation décrite par les autres arguments
            """
            x_transfo_ell, y_transfo_ell = transformation_elliptique(x, y, ell, angle, xc, yc)
            x_transfo, y_transfo = jnp.einsum('ij, jmp -> imp',transfo_grid,jnp.array([x_transfo_ell,y_transfo_ell]))
            Y = pression_integree(P500,x_transfo, y_transfo,P0,c_500,beta)
            return coeff_Y_pint*Y
        def Mock_Y_map(P500,spgrid,xc,yc,angle,ell,P0,c_500,beta,transfo_grid):
        	X_range, Y_range = spgrid
        	Y_map = Y(P500,X_range, Y_range, xc, yc, angle, ell,P0,c_500,beta,transfo_grid)
        	return Y_map

#Standard deviation de Planck en pixels
        fwhm_planck_arcmin = 7
        std_planck_arcmin = fwhm_planck_arcmin/(2*sqrt(2*log(2)))

        def PSF_Planck(theta500,x):
        	std_planck_pix = std_planck_arcmin*1024/(20*theta500)
        	return 1/(2*pi*std_planck_pix**2)*jnp.exp(-x**2/(2*std_planck_pix**2))

        def convol_Mock_Y_map(P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta,transfo_grid=jnp.array([[1,0],[0,1]])):
        	
        	# Grilles spatiale et fréquentielle
        	
        	mock=Mock_Y_map(P500,spgrid,xc,yc,angle,ell,P0,c_500,beta,transfo_grid)
        	convol_mock = jnp.real(jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fft2(mock)*jnp.fft.fft2(PSF))))
        	return convol_mock
        

#---------------------------------------------------

class mean_model:
	""" 
	Pour appeler rapidement un modèle moyen depuis n'importe quel programme. L'attribut healpix renvoie la carte rebin.
	Le nom de l'amas est une chaîne de caractères.
	"""
	
	def __init__(self,P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta):
		self.angle=angle
		self.ell=ell
		self.P0=P0
		self.c_500=c_500
		self.beta=beta
		self.mean_model_map=convol_Mock_Y_map(P500,PSF,spgrid,xc,yc,angle,ell,P0,c_500,beta)

	def healpix(self,amas,n_R500,savefile=False):
		image=self.mean_model_map
		
		savepath='mean_models/'+amas+'_mean_model_healpix_'+str(n_R500)+'R500.npy'
		
		if savefile:
		
			image_rebin=np.zeros(np.shape(image))
			PIXEL=pix_image(amas,n_R500)
			for h in np.unique(PIXEL):
				mask = PIXEL == h
				pixel_mean = np.mean(image[mask])
				image_rebin[mask] = pixel_mean
				
			with open(savepath,'wb') as f:
				jnp.save(f,image_rebin)
				
		else:
			if exists(savepath):
				with open(savepath,'rb') as f:
					image_rebin=jnp.load(f)
					
			else:
				image_rebin=np.zeros(np.shape(image))
				PIXEL=pix_image(amas,n_R500)
				for h in np.unique(PIXEL):
					mask = PIXEL == h
					pixel_mean = np.mean(image[mask])
					image_rebin[mask] = pixel_mean
		
		return image_rebin

class mean_model_3D:

    def __init__(self,P500,spgrid_3D,cube_size,L_size,xc,yc,angle,ell,P0,c_500,beta):
        spx, spy, spl = spgrid_3D
        spx_transfo, spy_transfo = transformation_elliptique_3D(spx, spy, ell, angle, xc, yc)
        # L'échelle est en R500 sur la ligne de visée pour faciliter  l'intégration.
        spl_transfo = spl - L_size//2*jnp.ones((cube_size,cube_size,L_size))[None,...] # Centrage sur la ligne de visée
        spl_transfo = pix_to_R500(spl_transfo)
        self.cube = coeff_Y_pint * pression(P500,spx_transfo,spy_transfo,spl_transfo,P0,c_500,beta)


