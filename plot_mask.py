import numpy as np
import healpy as hp
# import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy
import cPickle as pk
import gzip
from pylab import *

switch_backend('Agg')

# Params
radius = 10
nside  = 2048

#Files
mask_gal_file  = 'data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
mask_pts_file  = 'data/HFI_Mask_PointSrc_2048_R2.00.fits'
mask_cnt_file  = 'data/holes_mask_radius'+str(radius)+'deg_overlap0.3_nside'+str(nside)+'.pkl.gz'

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

# Running over patches
print('...running over patches...')
for i in xrange(len(x)):
	# Creating circular masks
	ipix = hp.query_disc(nside, (x[i], y[i], z[i]), np.radians(radius))
	mask[ipix] = 0.

hp.mollview(mask)
hp.graticule()
savefig('plots/gal_ptsrc_mask_holes_mask_radius'+str(nside)+'deg_overlap0.3_nside'+str(nside)+'.pdf')


