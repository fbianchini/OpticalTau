import numpy as np
import healpy as hp
import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy
import cPickle as pk
import gzip as gz

# Params
radius   = 15   # degree
nside    = 2048
fsky     = 0.1
max_iter = 3000
overlap  = 0.3
overlap_mask = 0.1


#Files
mask_gal_file = 'data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
mask_pts_file = 'data/HFI_Mask_PointSrc_2048_R2.00.fits'
output_file   = 'data/holes_mask_radius'+str(radius)+'deg_overlap'+str(overlap)+'_nside'+str(nside)+'.pkl.gz'

# Planck Galactic + PS Mask
print('...loading CMB masks...')
mask_gal = hp.reorder(fits.getdata(mask_gal_file, hdu=1)['GAL080'], inp='NESTED',out='RING') 
mask_pts = np.ones_like(mask_gal)
for f in ['F100', 'F143', 'F217', 'F353', 'F545', 'F857']:
	mask_pts *= hp.reorder(fits.getdata(mask_pts_file, hdu=1)[f], inp='NESTED',out='RING') 

mask = mask_gal * mask_pts
print('...done...')

# Creating holes
mask, (x,y,z) = hpy.ThrowCircleMasksOnTheSky(radius, 
											 mask=mask, 
											 nside=nside, 
											 overlap=overlap, 
											 overlap_mask=overlap_mask, 
											 fsky=fsky, 
											 max_iter=max_iter, 
											 return_pix=True)

output = {}
# output['mask'] = mask
output['xyz_circles'] = (x,y,z)
output['radius'] = radius
output['nside'] = nside
output['fsky'] = fsky
output['max_iter'] = max_iter
output['overlap'] = overlap
output['overlap_mask'] = overlap_mask

# Dumping to file
pk.dump(output, gz.open(output_file, 'wb'), protocol=2)