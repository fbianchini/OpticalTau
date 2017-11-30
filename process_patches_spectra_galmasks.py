import numpy as np
import cPickle as pk
import gzip as gz
from IPython import embed
import glob
from pylab import *
from astropy.visualization import hist
from scipy import stats
from cosmojo.universe import Cosmo

def GetDls(spectra_folder):
	patches_files = sorted(glob.glob('%s/*.pkl.gz' %(spectra_folder)))
	dls = []
	for f in patches_files:
		dls_tmp = np.asarray(pk.load(gz.open(f,'rb'))['dltt'])
		if len(dls_tmp) == 0:
			pass
		else:
			dls.append( dls_tmp )
	dls = np.vstack([dls[i] for i in xrange(len(dls))])
	return dls

def GetTau(tt, tt_ref, lmin=650, lmax=2000, method='RK'):
	assert (tt.size == tt_ref.size)
	delta_ell = lmin - lmax
	if method == 'RK' :
		return np.sum(-0.5 * np.log(tt[lmin:lmax+1]/tt_ref[lmin:lmax+1])) / delta_ell
	elif method == 'CR':
		return -0.5 * np.log(np.sum(tt[lmin:lmax+1])/np.sum(tt_ref[lmin:lmax+1]))# / delta_ell

def GetManyTaus(dls, dl_ref, lmin=650, lmax=2000, method='RK'):
	assert (dls.shape[1] == dl_ref.shape[0])
	taus = np.zeros(dls.shape[0])
	for i in xrange(dls.shape[0]):
		taus[i] = GetTau(dls[i,:], dl_ref, lmin=lmin, lmax=lmax, method=method)
	return taus

def err_skew(taus):
	n = len(taus)
	return np.sqrt(6.*n*(n-1)/((n-2)*(n+1)*(n+3)))

# Params
# lmin = 500
lmax = 1900

#Files
spectra_folder_10deg_gal080 = 'spectra_patches_radius10deg_gal080_MASTER'
spectra_folder_10deg_gal070 = 'spectra_patches_radius10deg_gal070'
spectra_ref_file_gal080     = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_MASTER.dat'
spectra_ref_file_gal070     = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal070.dat'

l, dltt_ref_gal070 = np.loadtxt(spectra_ref_file_gal070, unpack=1)
l, dltt_ref_gal080 = np.loadtxt(spectra_ref_file_gal080, unpack=1)

dls_10deg_gal070 = GetDls(spectra_folder_10deg_gal070)
dls_10deg_gal080 = GetDls(spectra_folder_10deg_gal080)

embed()

fig, ax = subplots(2,2, figsize=(15,10))

# Different lmax plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~
for iLM, LM in enumerate([1200,1900]):
	ax[iLM,0].set_title(r'Radius 10 deg - GAL070 - $\ell_{\rm max}=%d - N=%d$'%(LM,dls_10deg_gal070.shape[0]), size=15)
	ax[iLM,1].set_title(r'Radius 10 deg - GAL080 - $\ell_{\rm max}=%d - N=%d$'%(LM,dls_10deg_gal080.shape[0]), size=15)
	for lm in [300, 650, 1000]:
		taus_10_gal070 = GetManyTaus(dls_10deg_gal070, dltt_ref_gal070, lm, LM, method='CR')
		taus_10_gal080 = GetManyTaus(dls_10deg_gal080, dltt_ref_gal080, lm, LM, method='CR')

		hist(taus_10_gal070, 'knuth', histtype='step', ax=ax[iLM,0], label=r'$\ell_{\rm min} = %d\, \gamma_1=%.2f\pm%.2f$'%(lm,stats.skew(taus_10_gal070),err_skew(taus_10_gal070)))
		hist(taus_10_gal080, 'knuth', histtype='step', ax=ax[iLM,1], label=r'$\ell_{\rm min} = %d\, \gamma_1=%.2f\pm%.2f$'%(lm,stats.skew(taus_10_gal080),err_skew(taus_10_gal080)))

for i in xrange(2):
	ax[0,i].legend(loc='best')
	ax[1,i].legend(loc='best')
	ax[1,i].set_xlabel(r'$\hat{\tau}$', size=15)
	ax[0,i].set_xlim(-0.1,0.1)
	ax[1,i].set_xlim(-0.1,0.1)
	ax[0,i].axvline(0, ls='--', color='grey')
	ax[1,i].axvline(0, ls='--', color='grey')

# savefig('plots/taus_smica_halfmission1_cross_halfmission2_filt_200_2000_lmax1200_vs_lmax1900_galmasks_comparison_methodCR.pdf', bboxes_inches='tight')

show()











