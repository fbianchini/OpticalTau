import numpy as np
import healpy as hp
# import curvspec.master as cs
from astropy.io import fits
import hpyroutines.utils as hpy
import cPickle as pk
import gzip
from pylab import *
import sys
switch_backend('Agg')

"""
"""
glon, glat, radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) # QSO void center and radius [deg]


# Params
nside = 2048

#Files
mask_gal_file  = 'data/HFI_Mask_GalPlane-apo2_2048_R2.00.fits'
mask_pts_file  = 'data/HFI_Mask_PointSrc_2048_R2.00.fits'

# Mask
print('...loading CMB masks...')
mask_gal = hp.reorder(fits.getdata(mask_gal_file, hdu=1)['GAL080'], inp='NESTED',out='RING') 
mask_pts = np.ones_like(mask_gal)
for f in ['F100', 'F143', 'F217', 'F353', 'F545', 'F857']:
	mask_pts *= hp.reorder(fits.getdata(mask_pts_file, hdu=1)[f], inp='NESTED',out='RING') 

mask = mask_gal * mask_pts
print('...done...')

x, y, z = hp.ang2vec(np.radians(90.-glat),np.radians(glon))
ipix = hp.query_disc(nside, (x,y,z), np.radians(radius))
mask_tmp = np.zeros_like(mask)
mask_tmp[ipix] = 1.
mask_tmp = hp.smoothing(mask_tmp, fwhm=np.radians(30./60.))
mask_tmp *= mask.copy()

hp.mollview(mask_tmp)
hp.graticule()
verdic = {'-45,-45': r'\textbf{$-45^{0}$}','-45,-5.': r'\textbf{$0^{0}$}','-45,45': r'\textbf{$45^{0}$}', '45,15':'9h', '90,15':'6h', '135,15':'3h','0,15':'0h','-45,15':'21h','-90,15':'18h','-135,15':'15h'}

gratcolor = 'gainsboro'
for kk in verdic.keys():
	xloc, yloc = kk.split(',')
	if len(verdic[kk])==2:
		t=hp.projtext(float(xloc),float(yloc),verdic[kk],fontsize=10.,fontweight='bold',color=gratcolor,coord='C',lonlat=True,rotation = 90)
	else:
		t=hp.projtext(float(xloc),float(yloc),verdic[kk],fontsize=10.,fontweight='bold',color=gratcolor,coord='C',lonlat=True)
	#t.set_bbox(dict(facecolor=my_cmap(3),boxstyle='round', alpha=.2, edgecolor='None'))

savefig('plots/QSO_void_mask_gal080_radius'+str(radius)+'deg_eq_labels.pdf')


