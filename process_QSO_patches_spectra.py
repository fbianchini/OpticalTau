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

def GetTau(tt, tt_ref, l, lmin=650, lmax=2000, method='CR'):
	assert (tt.size == tt_ref.size)
	tt_ = tt[np.where((l>=lmin) & (l<=lmax))]
	tt_ref_ = tt_ref[np.where((l>=lmin) & (l<=lmax))]
	if method == 'RK' :
		delta_ell = lmin - lmax
		return np.sum(-0.5 * np.log(tt[lmin:lmax+1]/tt_ref[lmin:lmax+1])) / delta_ell
	elif method == 'CR':
		# return -0.5 * np.log(np.sum(tt[lmin:lmax+1])/np.sum(tt_ref[lmin:lmax+1]))
		return -0.5 * np.log(np.sum(tt_)/np.sum(tt_ref_))

def GetManyTaus(dls, dl_ref, l, lmin=650, lmax=2000, method='CR'):
	assert (dls.shape[1] == dl_ref.shape[0])
	taus = np.zeros(dls.shape[0])
	for i in xrange(dls.shape[0]):
		taus[i] = GetTau(dls[i,:], dl_ref, l, lmin=lmin, lmax=lmax, method=method)
	return taus

def err_skew(taus):
	n = len(taus)
	return np.sqrt(6.*n*(n-1)/((n-2)*(n+1)*(n+3)))

# Params
# lmin = 500
lmax = 1900

#Files
spectra_folder_5deg    = 'spectra_patches_radius5deg_gal080_MASTER_deltaell20'
spectra_folder_10deg   = 'spectra_patches_radius10deg_gal080_MASTER_deltaell20'
spectra_ref_file       = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_MASTER_deltaell20.dat'
spectra_QSO_5deg_file  = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_QSOvoid_radius5.0deg_MASTER_deltaell20.dat'
spectra_QSO_10deg_file = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_QSOvoid_radius10.0deg_MASTER_deltaell20.dat'
spectra_ref_file       = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_MASTER_deltaell20.dat'

l, dltt_ref       = np.loadtxt(spectra_ref_file, unpack=1)
l, dltt_QSO_5deg  = np.loadtxt(spectra_QSO_5deg_file, unpack=1)
l, dltt_QSO_10deg = np.loadtxt(spectra_QSO_10deg_file, unpack=1)

# embed()

dls_5deg  = GetDls(spectra_folder_5deg)
dls_10deg = GetDls(spectra_folder_10deg)
# dls_15deg = GetDls(spectra_folder_15deg)


fig, ax = subplots(2,3, figsize=(22,10))
fig.suptitle(r'DATA - Radius 5  deg - $N=%d$'%(dls_5deg.shape[0]), size=15)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min}=%d,\ell_{\rm max}=%d)$'%(lm,LM), size=15)
		taus_5  = GetManyTaus(dls_5deg, dltt_ref, l, lm, LM, method='CR')

		# Random patches
		hist(taus_5,  'knuth', histtype='step', ax=ax[iLM,ilm], label=r'$\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_5),err_skew(taus_5)))

		# QSO patch
		ax[iLM,ilm].axvline(GetTau(dltt_QSO_5deg, dltt_ref, l, lmin=lm, lmax=LM, method='CR'), label='QSO patch')

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='best')
		ax[i,j].set_xlabel(r'$\hat{\tau}$', size=15)
		ax[i,j].set_xlim(-0.1,0.1)
		ax[i,j].axvline(0, ls='--', color='grey')

savefig('plots/taus_smica_halfmission1_cross_halfmission2_filt_200_2000_lmax1200_vs_lmax1900_methodCR_5deg_MASTER_DATA_QSOvoid.pdf', bboxes_inches='tight')

# show()

fig, ax = subplots(2,3, figsize=(22,10))
fig.suptitle(r'DATA - Radius 10  deg - $N=%d$'%(dls_10deg.shape[0]), size=15)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min}=%d,\ell_{\rm max}=%d)$'%(lm,LM), size=15)
		taus_10  = GetManyTaus(dls_10deg, dltt_ref, l, lm, LM, method='CR')

		# Random patches
		hist(taus_10,  'knuth', histtype='step', ax=ax[iLM,ilm], label=r'$\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_10),err_skew(taus_10)))

		# QSO patch
		ax[iLM,ilm].axvline(GetTau(dltt_QSO_10deg, dltt_ref, l, lmin=lm, lmax=LM, method='CR'), label='QSO patch')

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='best')
		ax[i,j].set_xlabel(r'$\hat{\tau}$', size=15)
		ax[i,j].set_xlim(-0.1,0.1)
		ax[i,j].axvline(0, ls='--', color='grey')

savefig('plots/taus_smica_halfmission1_cross_halfmission2_filt_200_2000_lmax1200_vs_lmax1900_methodCR_10deg_MASTER_DATA_QSOvoid.pdf', bboxes_inches='tight')

# show()

embed()




