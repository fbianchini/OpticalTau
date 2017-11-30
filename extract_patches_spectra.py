import numpy as np
import healpy as hp
import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy
import cPickle as pk
import gzip, sys, os

start, end = int(sys.argv[1]), int(sys.argv[2]) 

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
radius = 10 # degree 
nside = 2048
fwhm_apo = 30. # arcmin
low_pass = 2000
hi_pass  = 200

# Creating the filter
filt = hpy.filter_bandpass_1d(lmax+1, hi_pass, low_pass, 50)

#Files
half_one_file  = 'data/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-1.fits'
half_two_file  = 'data/COM_CMB_IQU-smica-field-Int_2048_R2.01_halfmission-2.fits'
mask_gal_file  = 'data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
mask_pts_file  = 'data/HFI_Mask_PointSrc_2048_R2.00.fits'
mask_cnt_file  = 'data/holes_mask_radius'+str(radius)+'deg_overlap0.3_nside'+str(nside)+'.pkl.gz'
output_folder  = 'data/spectra_patches_radius'+str(radius)+'deg_gal080_MASTER_deltaell20/'
if not os.path.exists(output_folder): os.system('mkdir %s' %(output_folder))
output_file    = 'dltt_spectra_smica_halfmission1_cross_halfmission2_holes_mask_radius'+str(radius)+'deg_overlap0.3_nside'+str(nside)+'_'

# CMB maps
print('...loading CMB maps...')
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

# Read centers of the circular masks
print('...reading circular masks file...')
mask_cnt = pk.load(gzip.open(mask_cnt_file,'rb'))
print('...done...')

(x,y,z) = mask_cnt['xyz_circles'] 

results = {}
results['xyz_circles'] = (x,y,z)
results['dltt'] = []

# Running over patches
print('...running over patches...')
# for i in xrange(len(x)):
for i in xrange(start,end):
	if i < len(x):
		print('\tsim#%d' %i)

		# Creating circular masks
		ipix = hp.query_disc(nside, (x[i], y[i], z[i]), np.radians(radius))
		mask_tmp = np.zeros_like(mask)
		mask_tmp[ipix] = 1.
		mask_tmp = hp.smoothing(mask_tmp, fwhm=np.radians(fwhm_apo/60.))
		mask_tmp *= mask.copy()

		# Estimator
		est = cs.Master(mask_tmp, 
						lmin=lmin, 
					    lmax=lmax, 
					    delta_ell=delta_ell, 
					    MASTER=1, 
					    pixwin=True, 
					    flat=lambda l: l*(l+1)/2/np.pi, 
					    fwhm_smooth=fwhm)

		# Extract cross-power spectrum
		print('...extracting spectrum...')
		results['dltt'].append(est.get_spectra(half_one, map2=half_two))#, analytic_errors=True)
		# results['dltt'].append(GetSpectra(half_one, mask_tmp, lmax, map2=half_two, pixwin=True, beam=fwhm, flat=True))#, analytic_errors=True)

		del mask_tmp

	else:
		pass

print('...done...')

results['lb'] = est.lb
# results['lb'] = np.arange(lmax+1)

# Dumping sepctra to file
pk.dump(results, gzip.open(output_folder+output_file+'%d_%d.pkl.gz'%(start,end), 'wb'), protocol=2)
