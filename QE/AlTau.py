import numpy as np
import wignercoupling as wc
from cosmojo.universe import Cosmo
from cosmojo.utils import nl_cmb
from IPython import embed
from pylab import *
from numba import jit
from camb.bispectrum import threej

# @jit
def GetAlTau(cltt, nltt, lmax,):
    assert( cltt.size >= lmax+1)
    assert( nltt.size >= lmax+1)

    AlTau = np.zeros(lmax+1)

    for l in xrange(2,lmax+1):
        for lp in xrange(2,lmax+1):
            Lmin = np.abs(l-lp)
            Lmax = np.abs(l+lp)
            
            Lvals = np.arange(Lmin, Lmax+1)
            idx = Lvals <= lmax

            fact = cltt[l]**2 / ( (cltt[l]+nltt[l]) * (cltt[lp]+nltt[lp]) )
            fact *= (2*l+1)*(2*lp+1)/16./np.pi

            wig2 = threej(l,lp,0,0)**2
            # wig2 = np.asarray(wc.wigner3j_vect(2*l,2*lp,0.,0.)**2)
            
            AlTau[Lvals[idx]] += fact * wig2[idx]
                
    return AlTau#np.nan_to_num(1./AlTau)

def GetAlTauAtL(L, cltt, nltt, lmax,):
    assert( cltt.size >= lmax+1)
    assert( nltt.size >= lmax+1)

    AlTau = 0.#np.zeros(lmax+1)

    for l in xrange(2,lmax+1):
        for lp in xrange(2,lmax+1):
            # Lmin = np.abs(l-lp)
            # Lmax = np.abs(l+lp)
            
            # Lvals = np.arange(Lmin, Lmax+1)
            # idx = Lvals <= lmax

            fact = cltt[l]**2 / ( (cltt[l]+nltt[l]) + (cltt[lp]+nltt[lp]) )
            fact *= (2*l+1)*(2*lp+1)/16./np.pi

            wig2 = threej(l,lp,0,0)**2
            # wig2 = np.asarray(wc.wigner3j_vect(2*l,2*lp,0.,0.)**2)
            
            AlTau[Lvals[idx]] += fact * wig2[idx]
                
    return AlTau#np.nan_to_num(1./AlTau)

# @jit
# def GetAlTau(cltt, nltt, lmax,):
# 	assert( cltt.size >= lmax+1)
# 	assert( nltt.size >= lmax+1)

# 	AlTau = np.zeros(lmax+1)

# 	for L in xrange(2,lmax+1):
# 		for l in xrange(lmax+1):
# 			for lp in xrange(lmax+1):
# 				print L, l, lp
# 				fact = cltt[l]**2/((cltt[l]+nltt[l])+(cltt[lp]+nltt[lp]))
# 				fact *= (2*l+1)*(2*lp+1)/16./np.pi
# 				# embed()
# 				try:
# 					wig2 = wc.wigner3j(2*L,2*l,2*lp,0.,0.,0.)**2
# 				except:
# 					wig2 = 0.

# 				AlTau[L] += fact * wig2

# 	return np.nan_to_num(1./AlTau)

if __name__=='__main__':
	Delta_T = 3. # microK-arcmin
	fwhm    = 0. # arcmin
	lmax    = 500

	cosmo = Cosmo()
	cltt = cosmo.cmb_spectra(lmax)[:,0]
	nltt = nl_cmb(Delta_T, fwhm, lmax=lmax)

	norm = GetAlTau(cltt, nltt, lmax)

	embed()