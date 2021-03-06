import numpy as np
import healpy as hp
import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy

"""
This script extracts the CMB TT cross-power spectrum between the SMICA mission 1 x mission 2 maps
"""
def GetSpectra(map1, mask, lmax, map2=None, pixwin=True, beam=None, flat=True):
	if pixwin:
		pl = hp.pixwin(hp.npix2nside(len(map1)))[:lmax+1]
	else: 
		pl = np.ones(lmax+1)
	if beam is not None:
		bl = hp.gauss_beam(np.radians(beam/60.), lmax=lmax)
	else:
		bl = np.ones(lmax+1)
	
	fsky = np.mean(mask**2)

	cl = hp.anafast(map1*mask, map2=map2, lmax=lmax)

	l = np.arange(lmax+1)

	if flat:
		fact = l*(l+1)/2./np.pi
	else:
		fact = 1.

	return cl / fsky / bl**2 / pl**2 * fact

# Params
lmin = 2
lmax = 2500
fwhm = 5. # arcmin
delta_ell = 20
low_pass = 2000
hi_pass  = 200

# Creating the filter
filt = hpy.filter_bandpass_1d(lmax+1, hi_pass, low_pass, 50)

#Files
mask_gal_file  = 'data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
mask_pts_file  = 'data/HFI_Mask_PointSrc_2048_R2.00.fits'
half_one_file  = 'data/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-1.fits'
half_two_file  = 'data/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-2.fits'
output_cl_file = 'data/Dl_smica_halfmission1_cross_halfmission2_filt_'+str(hi_pass)+'_'+str(low_pass)+'_gal080_MASTER_deltaell20.dat'

# CMB maps
print('...loading and filtering CMB maps...')
half_one = hp.read_map(half_one_file)
half_two = hp.read_map(half_two_file)
half_one_alm = hp.map2alm(half_one, lmax=lmax)
half_two_alm = hp.map2alm(half_two, lmax=lmax)
half_one = hp.alm2map(hp.almxfl(half_one_alm, filt), hp.npix2nside(len(half_one)),lmax=lmax)
half_two = hp.alm2map(hp.almxfl(half_two_alm, filt), hp.npix2nside(len(half_two)),lmax=lmax)
print('...done...')

# Mask
print('...loading CMB masks...')
mask_gal = hp.reorder(fits.getdata(mask_gal_file, hdu=1)['GAL080'], inp='NESTED',out='RING') 
mask_pts = np.ones_like(mask_gal)
for f in ['F100', 'F143', 'F217', 'F353', 'F545', 'F857']:
	mask_pts *= hp.reorder(fits.getdata(mask_pts_file, hdu=1)[f], inp='NESTED',out='RING') 

mask = mask_gal * mask_pts
print('...done...')

# Estimator
est = cs.Master(mask, lmin=lmin, 
					  lmax=lmax, 
					  delta_ell=delta_ell, 
					  MASTER=1, 
					  pixwin=True, 
					  flat=lambda l: l*(l+1)/2/np.pi, 
					  fwhm_smooth=fwhm)

# Extract cross-power spectrum
print('...extracting spectrum...')
cl_tt = est.get_spectra(half_one, map2=half_two)#, analytic_errors=True)
# cl_tt = GetSpectra(half_one, mask, lmax, map2=half_two, pixwin=True, beam=fwhm, flat=True)
lb = est.lb
# lb = np.arange(lmax+1)
print('...done...')

np.savetxt(output_cl_file, np.c_[lb, cl_tt], header='ell D_l')





