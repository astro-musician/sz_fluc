#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: stagiaire

IMPORTANT : Le dossier data_planck doit être placé dans la même arborescence que le dossier contenant les programmes
qui utilisent ce package.

"""
from math import *
import numpy as np
import jax.numpy as jnp
# import jax.scipy.integrate as jsi
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle, Galactic
from astropy.wcs import WCS, FITSFixedWarning
from os.path import exists
import pickle
from .pathfinder import prog_path, tmpdir_path

prog_path=prog_path()
tmpdir_path='../'


# Passage des pixels en R500 :
def pix_to_R500(x):
    return x * 20. / 1024.
def R500_to_pix(x):
    return jnp.int32(x * 1024. / 20.)

def portion(image,xc,yc,n):
    
    r500_pix= 1024/20 # Le rescaling assure cette conversion
    side_length = n*r500_pix
    row_min = int(xc-side_length)
    col_min = int(yc-side_length)
    row_max = int(xc+side_length)
    col_max = int(yc+side_length)
    
    reduced=image[row_min:row_max,col_min:col_max]
    return jnp.array(reduced)
    

def reduce_n_R500(image,n):
    """
    Réduit l'image en un carré de côté n R500 autour du centre
    """
    r500_pix= 1024/20 # Le rescaling assure cette conversion
    side_length = n*r500_pix
    row_min = int(512-side_length)
    col_min = int(512-side_length)
    row_max = int(512+side_length)
    col_max = int(512+side_length)
    reduced=image[row_min:row_max,col_min:col_max]
    return jnp.array(reduced)

def pix_shape(n_R500):
    im=jnp.zeros((1024,1024))
    return jnp.shape(reduce_n_R500(im,n_R500))

def spnorm(n_R500):
    n_cols, n_rows = pix_shape(n_R500)
    spx = jnp.arange(n_cols) - jnp.ones(n_cols) * (n_cols // 2)
    spy = jnp.arange(n_rows) - jnp.ones(n_rows) * (n_rows // 2)
    spgrid = jnp.meshgrid(spx, spy)
    return jnp.sqrt(spgrid[0] ** 2 + spgrid[1] ** 2)

def spnorm2(n_R500):
    n_cols, n_rows = R500_to_pix(n_R500), R500_to_pix(n_R500)
    spx = jnp.arange(n_cols) - jnp.ones(n_cols) * (n_cols // 2)
    spy = jnp.arange(n_rows) - jnp.ones(n_rows) * (n_rows // 2)
    spgrid = jnp.meshgrid(spx, spy)
    return jnp.sqrt(spgrid[0] ** 2 + spgrid[1] ** 2)

def spnorm_3D(n_R500):
    len_x, len_y, len_z = pix_shape(n_R500)[0], pix_shape(n_R500)[0], pix_shape(5)[0]
    spx = jnp.arange(len_x) - jnp.ones(len_x) * (len_x // 2)
    spy = jnp.arange(len_y) - jnp.ones(len_y) * (len_y // 2)
    spz = jnp.arange(len_z) - jnp.ones(len_z) * (len_z // 2)
    spgrid = jnp.meshgrid(spx, spy, spz)
    return jnp.sqrt(spgrid[0] ** 2 + spgrid[1] ** 2 + spgrid[2]**2)

def knorm(n_R500):
    n_cols, n_rows = pix_shape(n_R500)
    kfreq_x = jnp.fft.fftfreq(n_cols)
    kfreq_y = jnp.fft.fftfreq(n_rows)
    kgrid = jnp.meshgrid(kfreq_x, kfreq_y)
    return jnp.sqrt(kgrid[0] ** 2 + kgrid[1] ** 2)

def knorm_3D(n_R500):
    len_x, len_y, len_z = pix_shape(n_R500)[0], pix_shape(n_R500)[0], pix_shape(5)[0]
    kfreq_x, kfreq_y, kfreq_z = jnp.fft.fftfreq(len_x), jnp.fft.fftfreq(len_y), jnp.fft.fftfreq(len_z)
    kgrid = jnp.meshgrid(kfreq_x, kfreq_y, kfreq_z)
    return jnp.sqrt(kgrid[0]**2 + kgrid[1]**2 + kgrid[2]**2)

def spgrid(n_R500):
    im_size_x, im_size_y = pix_shape(n_R500)
    x_range = jnp.arange(im_size_x)
    y_range = jnp.arange(im_size_y)
    return jnp.meshgrid(x_range,y_range,indexing='ij')

def spgrid_3D(n_R500):
    im_size_x, im_size_y, im_size_z = pix_shape(n_R500)[0], pix_shape(n_R500)[0], pix_shape(5)[0]
    x_range = jnp.arange(im_size_x)
    y_range = jnp.arange(im_size_y)
    z_range = jnp.arange(im_size_z)
    return jnp.meshgrid(x_range,y_range,z_range,indexing='ij')

def spgrid_3D_transfo(amas):

    with open(tmpdir_path+'transfert/' + amas + '_transfo_spgrid.pkl', 'rb') as f:
        transfo_dict = pickle.load(f)

    transfo_spgrid = transfo_dict['transfo_spgrid']

    side_x, side_y = jnp.shape(transfo_spgrid[0])
    mesh_x = jnp.ones((side_x, side_y, pix_shape(5)[0])) * transfo_spgrid[0][...,None]
    mesh_y = jnp.ones((side_x, side_y, pix_shape(5)[0])) * transfo_spgrid[1][...,None]
    mesh_z = jnp.ones((side_x, side_y, pix_shape(5)[0])) * jnp.arange(pix_shape(5)[0])[None,None,...]
    return jnp.float32([mesh_x,mesh_y,mesh_z])

def knorm_3D_transfo(amas):

    with open(tmpdir_path+'transfert/' + amas + '_transfo_spgrid.pkl', 'rb') as f:
        transfo_dict = pickle.load(f)

    transfo_kgrid = transfo_dict['transfo_kgrid']

    side_x, side_y = jnp.shape(transfo_kgrid[0])
    mesh_kx = jnp.ones((side_x, side_y, pix_shape(5)[0])) * transfo_kgrid[0][..., None]
    mesh_ky = jnp.ones((side_x, side_y, pix_shape(5)[0])) * transfo_kgrid[1][..., None]
    len_kz = pix_shape(5)[0]
    kz = jnp.fft.fftfreq(len_kz)
    mesh_kz = jnp.ones((side_x, side_y, pix_shape(5)[0])) * kz[None,None,...]
    return jnp.sqrt(mesh_kx**2+mesh_ky**2+mesh_kz**2)
    
#---------------------------------------------------
# ECHANTILLONAGE HEALPIX
#---------------------------------------------------

def image(amas,n_R500):
    imagepath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
    full_image=jnp.float32(fits.open(imagepath)[0].data)
    return reduce_n_R500(full_image,n_R500)

def image_for_noise(amas,xc,yc,n_R500):
    return portion(image(amas,10),xc,yc,n_R500)

def header(amas):
    imagepath = tmpdir_path+'data_planck/maps-DR2015-7arcmin/' + amas + '_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
    return fits.open(imagepath)[0].header
    
def pix_image(amas,n_R500):
	pixpath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_PIXEL_INDEX_2048_MILCA.fits'
	full_pix_image = jnp.int32(fits.open(pixpath)[0].data)
	return reduce_n_R500(full_pix_image, n_R500)

def pix_image_noise(amas,xc,yc,n_R500):
    return portion(pix_image(amas,10), xc, yc, n_R500)

def healpix_mean_side(amas):
    return 5

def mm_bfit(amas,n_R500):
    mm_path=tmpdir_path+'mean_models/'+amas+'_bestfit.fits'
    full_mm = jnp.float32(fits.open(mm_path)[0].data)
    return reduce_n_R500(full_mm,n_R500)

def fluc(amas,n_R500):
    fluc_path=tmpdir_path+'fluc_maps/'+amas+'_fluc_map.fits'
    full_fluc_image = jnp.float32(fits.open(fluc_path)[0].data)
    return reduce_n_R500(full_fluc_image,n_R500)

def rebin(image,new_space_sample):
    size=image.shape[0]
    new_shape = (size // new_space_sample, size // new_space_sample)
    image = image[:size - size % new_space_sample, :size - size % new_space_sample]
    shape = (new_shape[0], image.shape[0] // new_shape[0], new_shape[1], image.shape[1] // new_shape[1])
    return image.reshape(shape).mean(-1).mean(1)

def rebin_3D(rect,new_space_sample):
    size=rect.shape[0]
    z_size=rect.shape[2]
    new_shape = (size // new_space_sample, size // new_space_sample)
    rect = rect[:size - size % new_space_sample, :size - size % new_space_sample,:]
    shape = (new_shape[0], rect.shape[0] // new_shape[0], new_shape[1], rect.shape[1] // new_shape[1], z_size)
    return rect.reshape(shape).mean(3).mean(1)

def spgrid_rebin(n_R500,space_sampling):
    base_spgrid=spgrid(n_R500)
    spgrid_x=rebin(base_spgrid[0],space_sampling)
    spgrid_y = rebin(base_spgrid[1], space_sampling)
    return jnp.array([spgrid_x,spgrid_y])

def spnorm_rebin(n_R500,space_sampling):
    return rebin(spnorm(n_R500),space_sampling)


#---------------------------------------------------
# RÉCUPÉRATION DES INFOS SUR L'AMAS DEPUIS LA MASTER TABLE 
#---------------------------------------------------

masterfilepath = tmpdir_path+'data_planck/XCOP_master_table.fits'
master=Table.read(masterfilepath)

def numero_amas(amas):
    """
    Renvoie le numéro de l'amas name dans la master table X-COP
    amas : str
    """
    #master = master_table(amas)
    ind=0
    NAME = master['NAME']
    table_length = len(NAME)
    while ind < table_length:
        if NAME[ind] == amas:
            num_amas = ind
            return ind
        else:
            ind +=1
    return None

def M500(amas):
    """
    Donne la valeur M500 (x1e14 Msun) pour l'amas depuis la table X-COP
    amas : str
    """
    return master['M500_tot'][numero_amas(amas)]

def redshift(amas):
    """
    Donne le redshift de l'amas
    """
    return master['REDSHIFT'][numero_amas(amas)]

def theta500(amas):
	return master['Theta500'][numero_amas(amas)]

def RA(amas):
	return master['RA'][numero_amas(amas)]

def DEC(amas):
	return master['DEC'][numero_amas(amas)]

# Identificaion du centre de l'amas (avec rescaling du au fait que l'image est tronquée à nR500)

def skycoord(amas):
    return SkyCoord(RA(amas),DEC(amas),frame='icrs',unit='deg').transform_to(Galactic())
def center_coord_icrs(amas):
	filepath=tmpdir_path+'data_planck/maps-DR2015-7arcmin/'+amas+'_y_map_MILCA_070_to_857_GHz_7arcmin.fits'
	hdul = fits.open(filepath)
	header = hdul[0].header
	wcs = WCS(header)
	return jnp.int32(wcs.world_to_pixel(SkyCoord(RA(amas),DEC(amas),frame='icrs',unit='deg')))
	
def center_coord(amas,n_R500):
	xc,yc=center_coord_icrs(amas)
	xc, yc = xc - (512.-R500_to_pix(n_R500)), yc - (512.-R500_to_pix(n_R500))
	return jnp.array([xc,yc])

# Constantes physiques et paramètres Lambda CDM
Mpc = 3.086e22 # Mpc en mètres
c_km = 3e5 # vitesse de la lumière en km/s, utile dans les calculs impliquant H0
H0 = 70
Omega_M = 0.3

def P500(amas):
	E_z = jnp.sqrt(Omega_M*(1 + redshift(amas))**3 + 1 - Omega_M) # Rapport H(z)/H0
	d_ang = c_km*redshift(amas)/H0 # Distance angulaire à bas redshift
	R500 = Mpc*d_ang*theta500(amas)*pi/(180*60) # avec le facteur de conversion d'arcmin vers rad
	P500 = 3.426*1.6e-19*1e6*(M500(amas)/10)**(2/3)*E_z**(8/3)*R500
	return P500
    
