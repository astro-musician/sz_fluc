#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:34:14 2024

@author: stagiaire
"""

from math import *
import numpy as np
import jax.numpy as jnp
from jax.numpy import array as Array
from scipy.special import gamma
from scipy.stats import linregress
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from os.path import exists
import re
import pickle
from .pathfinder import prog_path, tmpdir_path
from .cluster_info import reduce_n_R500, spnorm, knorm, mm_bfit, header, fluc, theta500, healpix_mean_side

prog_path=prog_path()
tmpdir_path='../'

# Passage des pixels en R500 :

def pix_to_R500(x):
    return x*20./1024.

def R500_to_pix(x):
    return x*1024./20.

def truncate(amas,im,gal,radius):
    """
    Tronque im du disque de centre gal (degrés) et de rayon radius (arcmin)
    """
    #trunc_im = copy.deepcopy(im)
    filepath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
    hdul = fits.open(filepath)
    header = hdul[0].header
    wcs = WCS(header)

    n_cols, n_rows = np.shape(im)

    trunc_im = im
    x_center, y_center= gal
    x_center, y_center = np.intc(wcs.world_to_pixel(SkyCoord(x_center,y_center,frame='galactic',unit='deg')))
    radius_pix = radius*1024/20/theta500(amas)
    row_min = max(0,int(x_center-radius_pix))
    col_min = max(0,int(y_center-radius_pix))
    row_max = min(int(x_center+radius_pix),n_rows-1)
    col_max = min(int(y_center+radius_pix),n_cols-1)
    for x in range(row_min,row_max):
        for y in range(col_min,col_max):
            if (x-x_center)**2+(y-y_center)**2 <= radius_pix**2:
                trunc_im[x,y]=0
    return #trunc_im

# Cette fonction est à appliquer au masque pour chaque source ponctuelle (d'où l'absence de deepcopy)

#---------------------------------------------------
# MASQUE DU CENTRE : TRONCATURE CIRCULAIRE
#---------------------------------------------------

def mask_SZ(amas):

    mask_R500 = 5 #taille du masque autour du centre
    mask = np.ones((1024,1024))

    r500_pix= 1024/20 # Le rescaling assure cette conversion
    side_length = mask_R500*r500_pix

    mask[np.where(spnorm(10)<R500_to_pix(5))]=0

    noise_zone = np.where(mask==0)

# LECTURE DES FICHIERS RÉGIONS

    region_path = tmpdir_path+'data_planck/fichiers regions/'+amas+'.reg'
    with open(region_path,'r') as file:
        lignes = file.readlines()
        coordonnees_cercles = []
        for ligne in lignes:
            # Utiliser une expression régulière pour extraire les coordonnées des cercles
            match = re.match(r'circle\(([^,]+),([^,]+),([^"]+)"\)', ligne)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                center = np.array([x,y])
                rayon = float(match.group(3))/60
                coordonnees_cercles.append((x, y, rayon))
                truncate(amas,mask,center,rayon)

    for source in coordonnees_cercles:
        xc,yc,rad=source
        truncate(amas,mask,np.array([xc,yc]),rad)

    return mask

#------------------------------------------
# SIMULATION DE POWER SPECTRUM
#------------------------------------------

def mock_power_spectrum(slope,k_min,norm):
    """
    :param slope: pente logaritmique
    :param k_min: fréquence minimale de coupure (échelle d'injection pour la turbulence)
    :param norm: normalisation, identifié aux fluctuations de pression
    :return: array 2D avec les fréquences et les valeurs de spectre de puissance correspondantes
    """
    K_amp = np.geomspace(1/R500_to_pix(5),1/R500_to_pix(0.05),500)
    amp = norm*np.exp(-k_min/K_amp)*K_amp**(-slope)
    return np.array([K_amp,amp])

#------------------------------------------
# FILTRE MEXICAN HAT
#------------------------------------------

def gaussian(x,std):
    return 1/(2*pi*std[...,None,None]**2)*jnp.exp(-x[None,...]**2/(2*std[...,None,None]**2))
def gaussian_blur(image,std,spnorm):
    """
    Convolue l'image avec une gaussienne d'écart type spatial std pixels
    """
    return jnp.real(jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fft2(image[None,:,:])*jnp.fft.fft2(gaussian(spnorm,std))),axes=(1,2)))

def gaussian_blurs(images,FT_filters):
    """
    shape : ( num image , std , x , y )
    """
    return jnp.real(jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fft2(images[:, None, :, :]) * FT_filters),axes=(-2, -1)))
def convol_mexican_hat(image,mask,std,spnorm):
    """
    Convolue l'image et son masque avec un filtre mexican hat de parametres donnes dans l'espace des frequences
    """
    eps=1e-3
    std1=std/sqrt(1+eps)
    std2=std*sqrt(1+eps)
    k_cutoff=1e-10
    return (gaussian_blur(image, std1,spnorm)/jnp.clip(gaussian_blur(mask, std1,spnorm),a_min=k_cutoff, a_max=jnp.inf )- gaussian_blur(image, std2,spnorm)/jnp.clip(gaussian_blur(mask, std2,spnorm),a_min=k_cutoff, a_max=jnp.inf))*mask[None,:,:]

def cube_mexican_hat(images,mask,FT_filters1,FT_filters2):
    k_cutoff = 1e-10
    return (gaussian_blurs(images, FT_filters1) / jnp.clip(gaussian_blurs(mask[None,None,...], FT_filters1), a_min=k_cutoff,a_max=jnp.inf) -
            gaussian_blurs(images, FT_filters2) / jnp.clip(gaussian_blurs(mask[None,None,...], FT_filters2), a_min=k_cutoff, a_max=jnp.inf)) * mask[None,None, :, :]

#------------------------------------------
# TEST DE POWER SPECTRUM AVEC MASQUE DE LA RÉGION CENTRALE
#------------------------------------------


def power_spectrum(image,spnorm,mask,k_min,k_max,slope=2):
    """
    Renvoie le vecteur des fréquences et les valeurs de spectre de puissance correspondantes
    """

    n_spectral_rings = 10 # Nombre d'anneaux dans le rendu du spectre de puissance
    K_amp = jnp.geomspace(k_min,k_max,n_spectral_rings)

    std_reel = 1/(sqrt(2)*pi*K_amp)
    amp = jnp.var(convol_mexican_hat(image, mask, std_reel,spnorm),axis=(1,2),where=mask==1)
    amp = amp * 1e6 / (pi * K_amp ** 2) * jnp.shape(image)[0]*jnp.shape(image)[1] / jnp.sum(mask)

    # Correction du biais du au mexican hat (slope est l'opposé de la pente log)
    X=np.log(K_amp)[2:]
    Y=np.log(amp)[2:]
    slope=-linregress(X,Y)[0]
    bias = 2**(slope/2)*gamma(3-slope/2)/gamma(3)
    amp=amp#/bias

    return jnp.array([K_amp,amp])

def power_spectrum_mock_fluc(images,spnorm,mask,k_min,k_max,slopes):
    n_spectral_rings = 10
    eps=1e-3
    K_amp = jnp.geomspace(k_min,k_max,n_spectral_rings)
    std_reel = 1/(sqrt(2)*pi*K_amp)
    stds1 = std_reel / sqrt(1 + eps)
    stds2 = std_reel*sqrt(1 + eps)
    FT_filters1 = jnp.fft.fft2(gaussian(spnorm, stds1))
    FT_filters2 = jnp.fft.fft2(gaussian(spnorm, stds2))

    amps = jnp.var( cube_mexican_hat(images,mask,FT_filters1,FT_filters2), axis=(-2,-1), where=mask==1 )
    amps = amps / (eps**2 * pi * K_amp ** 2) * jnp.shape(mask)[0]*jnp.shape(mask)[1] / jnp.sum(mask)

    # Correction du biais du au mexican hat (slope est l'opposé de la pente log)
    bias = 2 ** (slopes / 2) * gamma(3 - slopes / 2) / gamma(3)
    amps = amps / bias[None,:,None]

    return amps


def power_spectrum_grid(image,spnorm,knorm,mask,k_min,k_max):

    K_amp, amp = power_spectrum(image,spnorm,mask,k_min,k_max)

    diff = np.abs(knorm[None,...]-K_amp[...,None,None])
    ind_pix, grid_x, grid_y = np.where(diff==np.min(diff,axis=0))

    # On découpe la grille de fréquences en anneaux autour des normes utilisées pour le calcul du PS (K_amp) :
    ind_Kamp_grid = np.zeros(np.shape(knorm))
    ind_Kamp_grid[grid_x,grid_y]=ind_pix
    PSgrid=amp[np.intc(ind_Kamp_grid)] # Grille du power spectrum

    return PSgrid


# L'architecture en boucles if imbriquées sert juste à la sauvegarde des fichiers.
# Il vaut quand même mieux les importer avant de s'en servir dans une boucle, plutôt que de les appeler à chaque itération.

class PS_noise_amas:

    def __init__(self,amas):
        filepath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
        jkpath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_JK_map_MILCA_070_to_857_GHz_7arcmin.fits'
        image=np.float32(fits.open(filepath)[0].data)
        jk_image=np.float32(fits.open(jkpath)[0].data)
        self.amas=amas
        self.y_map=image
        self.jk_map=jk_image
        self.mask=mask_SZ(amas)

    def PS(self,savefile=False,k_min=1/R500_to_pix(5),k_max=1/10):
        """
        Le vecteur de fréquences est en pixels^-1. Utiliser les fonctions de conversion pour l'avoir en R500^-1.
        savefile=True overwrite un power spectrum s'il existe déjà.
        """

        savepath=tmpdir_path+'power_spectra/'+self.amas+'_PS_noise.npy'

        if savefile:
            PS=power_spectrum(self.y_map,spnorm(10),self.mask,k_min=k_min,k_max=k_max)
            with open(savepath,'wb') as f:
                    np.save(f,PS)

        else:

            if exists(savepath):
                with open(savepath,'rb') as f:
                    PS = np.load(f)
            else:
                PS=power_spectrum(self.y_map,spnorm(10),self.mask,k_min,k_max)

        return PS

    def PS_grid(self,n_R500,savefile=False):

        k_min = 1/R500_to_pix(5)
        k_max = 1/(2*healpix_mean_side(self.amas))
        savepath=tmpdir_path+'power_spectra/'+self.amas+'_PS_noise_grid_'+str(n_R500)+'R500.npy'

        if savefile:

            PS_grid=power_spectrum_grid(self.y_map,spnorm(10),knorm(n_R500),self.mask,k_min,k_max)
            with open(savepath,'wb') as f:
                np.save(f,PS_grid)

        else:

            if exists(savepath):
                with open(savepath,'rb') as f:
                    PS_grid=np.load(f)

            else:
                PS_grid=power_spectrum_grid(self.y_map,spnorm(10),knorm(n_R500),self.mask,k_min,k_max)

        return PS_grid


# Rajout d'une classe pour le power spectrum des fluctuations, en utilisant le mean model

class PS_fluc_amas:

    def __init__(self,amas,n_R500,mask,k_min,k_max):
        self.amas=amas
        self.n_R500=n_R500
        self.fluc_map=fluc(self.amas,self.n_R500)
        self.spnorm=spnorm(n_R500)
        self.knorm=knorm(n_R500)
        self.k_min=k_min
        self.k_max=k_max

    def PS(self,savefile=False):
        savepath=tmpdir_path+'fluc/'+self.amas+'_PS_fluc.npy'

        if savefile:
            PS=power_spectrum(self.fluc_map,self.spnorm,mask=np.ones(np.shape(self.fluc_map)),k_min=self.k_min,k_max=self.k_max)
            with open(savepath,'wb') as f:
                np.save(f,PS)

        else:
            if exists(savepath):
                with open(savepath,'rb') as f:
                    PS = np.load(f)
            else:
                PS=power_spectrum(self.fluc_map,self.spnorm,mask=np.ones(np.shape(self.fluc_map)),k_min=self.k_min,k_max=self.k_max)

        return PS

    def PS_grid(self,savefile=False):
        k_min = 1/R500_to_pix(self.n_R500*sqrt(2))
        k_max = 1/(2*healpix_mean_side(self.amas))
        savepath=tmpdir_path+'fluc/'+self.amas+'_PS_grid_fluc.npy'

        if savefile:
            PS_grid=power_spectrum_grid(self.fluc_map,self.spnorm,self.knorm,mask=np.ones(np.shape(self.fluc_map)),k_min=k_min,k_max=k_max)
            with open(savepath,'wb') as f:
                np.save(f,PS_grid)

        else:
            if exists(savepath):
                with open(savepath,'rb') as f:
                    PS_grid = np.load(f)
            else:
                PS_grid=power_spectrum_grid(self.fluc_map,self.spnorm,self.knorm,mask=np.ones(np.shape(self.fluc_map)),k_min=k_min,k_max=k_max)

        return PS_grid