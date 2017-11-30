import numpy as np
import healpy as hp

def GetTauMap(tmap, cltt, nltt, nside=None, lmax=None, mmax=None, fwhm=0., pixwin=False):
	if nside is None:
		nside = hp.npix2nside(tmap.size)
	if lmax is None:
		lmax = 2*nside

	assert( len(cltt) >= lmax+1)
	assert( len(nltt) >= lmax+1)

	talm = hp.map2alm(tmap, lmax=lmax, mmax=mmax)

	if fwhm != 0.:
		bl = hp.gauss_beam(np.radians(fwhm/60.), lmax=lmax)
		talm = hp.almxfl(talm, bl, mmax=mmax)

	fl1 = cltt/(cltt+nltt) # probably nltt has to be divided by bl
	fl2 = 1./(cltt+nltt)   # probably nltt has to be divided by bl
	fl1[:2] = 0.
	fl2[:2] = 0.

	alm1 = hp.almxfl(talm, fl1, mmax=mmax)
	alm2 = hp.almxfl(talm, fl2, mmax=mmax)

	return hp.alm2map(alm1, nside, lmax=lmax, mmax=mmax, pixwin=pixwin, fwhm=fwhm) * hp.alm2map(alm2, nside, lmax=lmax, mmax=mmax, pixwin=pixwin, fwhm=fwhm) 

def GetTauAlm(tmap, cltt, nltt, nside=None, lmax=None, mmax=None, fwhm=0., pixwin=False):
	taumap = GetTauMap(tmap, cltt, nltt, nside=nside, lmax=lmax, mmax=mmax, fwhm=fwhm, pixwin=pixwin)
	return hp.map2alm(taumap, lmax=lmax, mmax=mmax)