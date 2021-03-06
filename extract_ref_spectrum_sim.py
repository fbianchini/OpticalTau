import numpy as np
import healpy as hp
import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy
import sys

"""
This script extracts the CMB TT of a simulated map
"""
wnoise = int(sys.argv[1])

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

	if flat:
		l = np.arange(lmax+1)
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
sim_tt_file    = 'data/tt_only_nopixwin_nside2048.fits'
sim_ntt1_file  = 'data/tt_noise1_nopixwin_nside2048.fits'
sim_ntt2_file  = 'data/tt_noise2_nopixwin_nside2048.fits'
if wnoise:
	output_cl_file  = 'data/Dl_sim_wnoise_nopixwin_beam_ptsrc_MASTER_delta20.dat'
else:
	# output_cl_file  = 'data/Dl_sim_wonoise_nopixwin_nobeam_nofilt_ptsrc_MASTER.dat'
	output_cl_file  = 'data/Dl_sim_wonoise_nopixwin_beam_ptsrc_MASTER_delta20.dat'

# CMB maps
print('...loading CMB maps...')
sim_tt1 = hp.read_map(sim_tt_file)
sim_tt1 = hp.smoothing(sim_tt1, fwhm=np.radians(fwhm/60.)) # Apply beam to signal-only map
if wnoise:
	sim_ntt1 = hp.read_map(sim_ntt1_file)
	sim_ntt2 = hp.read_map(sim_ntt2_file)
	sim_tt1  = sim_tt1 + sim_ntt1
	sim_tt2  = sim_tt1 + sim_ntt2

sim_tt1_alm = hp.map2alm(sim_tt1, lmax=lmax)
sim_tt1 = hp.alm2map(hp.almxfl(sim_tt1_alm, filt), hp.npix2nside(len(sim_tt1)),lmax=lmax)
if wnoise:
	sim_tt2_alm = hp.map2alm(sim_tt2, lmax=lmax)
	sim_tt2 = hp.alm2map(hp.almxfl(sim_tt2_alm, filt), hp.npix2nside(len(sim_tt2)),lmax=lmax)
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
					  pixwin=0, 
					  flat=lambda l: l*(l+1)/2/np.pi, 
					  fwhm_smooth=fwhm)

# Extract cross-power spectrum
print('...extracting spectrum...')
if wnoise:
	cl_tt = est.get_spectra(sim_tt1, map2=sim_tt2)
else:
	cl_tt = est.get_spectra(sim_tt1)

# cl_tt = est.get_spectra(sim_tt)#, analytic_errors=True)
# cl_tt = GetSpectra(sim_tt, mask, lmax, pixwin=True, beam=fwhm, flat=True)
print('...done...')
lb = est.lb
# lb = np.arange(lmax+1)

np.savetxt(output_cl_file, np.c_[lb, cl_tt], header='ell D_l')





