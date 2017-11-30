import numpy as np 
from pylab import *
import healpy as hp
import glob, os, sys
from astropy.io import fits
from scipy import ndimage
from IPython import embed

def fn_get_psource_mask(srccords,ip_coords=0,op_cords=0,ip_sources_in_ABS=1,radius=1.,nside=256,):

		#output map realted stuff
		nopixels=12*(nside)**2
		#pix=0.05; np.degrees(H.nside2resol(nside))
		pix=0.01

		if not ip_sources_in_ABS: #then some ABS map must be read here
				#abs-inthe location
				mapfile='/home/sri/analysis/2015_09/psource_masks/data/fs-cmb-fieldc-v58/cumulative/data_abs_data_fielda.fits' #field-c
				mapfile='/home/sri/analysis/2015_09/psource_masks/data/fs-cmb-fielda-v62/cumulative/data_abs_data_fielda.fits' #field-a

				#smoothing_fwhm=np.radians(1.)#32./60.)
				ASBUNMASK,ABSMASK,I,Q,U=fn_get_abs_map(mapfile)#,smoothing_fwhm=smoothing_fwhm)

		if ip_coords==0:
				pra,pdec=srccords['x'],srccords['y']
				plon,plat=hp.rotator.euler(pra,pdec,1)
		else:
				plon,plat=srccords['x'],srccords['y']
				pra,pdec=hp.rotator.euler(plon,plat,2)

		noofsources=len(pra)

		#OP MASK
		minx,maxx=-180.,180.
		miny,maxy=-90.,90.

		xxx=np.arange(minx,maxx+pix,pix)
		yyy=np.arange(miny,maxy+pix,pix)
		X,Y=np.meshgrid(xxx,yyy)

		#plot(pra,pdec,'ro');show();quit()

		#creating a hanning kernel
		npix_cos=int(radius/pix)+1
		hanning=np.hanning(npix_cos)
		hanning=np.sqrt(outer(hanning,hanning))
		# print np.shape(hanning)
		#imshow(hanning);colorbar();show();quit()

		MASK=np.ones(np.shape(X))
		
		cx,cy=plon,plat
		opcoordname='gal'

		cx=[val-360. if val>180. else val for val in cx]

		lll=0
		for i,j in zip(cx,cy):
				inds=np.where((X-i)**2. + (Y-j)**2. <= radius**2.)
				MASK[inds]=0.
				lll+=1

		# imshow(MASK);colorbar();show();quit()

		##apodizations tuff
		apodMASK=ndimage.convolve(MASK, hanning)
		apodMASK/=apodMASK.max()

		#ax=subplot(111);imshow(apodMASK);colorbar();show();quit()

		HMASK=np.zeros(nopixels)
		HHIT=np.zeros(nopixels)
		for r in range(np.shape(X)[0]):
				PP=hp.pixelfunc.ang2pix(nside,np.radians(90.-Y[r,:]),np.radians(X[r,:]))
				HMASK[PP]+=apodMASK[r,:]
				HHIT[PP]+=1
		HMASK[HHIT>0.]/=HHIT[HHIT>0.]
		#HMASK[ABSMASK]=fill_value

		# HMASKdic[opcoordname]=HMASK


		return HMASK


# Params
nside = 1024
radius = 10. # arcmin

# Files 'n' path
pccs_path = '/global/cscratch1/sd/fbianc/WxS_XC/data/PLANCK/PCCS/'#'/Users/fbianchini/Research/Data/COM_PCCS-Catalogs_vPR2/PCCS/'

srccords = {}
glon = []
glat = []

for f in glob.glob(pccs_path+'*.fits')[:2]:
	print f
	pcc = fits.getdata(f, hdu=1)
	glon.append( pcc.GLON )
	glat.append( pcc.GLAT )

inds = np.random.choice(np.arange(1000),10)

srccords['x'] = np.concatenate(glon)[inds]
srccords['y'] = np.concatenate(glat)[inds]

ptsrc_mask = fn_get_psource_mask(srccords,ip_coords=1,op_cords=1,ip_sources_in_ABS=1,radius=radius/60.,nside=nside)

# embed()






