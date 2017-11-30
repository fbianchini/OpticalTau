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
spectra_folder_5deg  = 'spectra_patches_radius5deg_gal080'
spectra_folder_10deg = 'spectra_patches_radius10deg_gal080'
spectra_folder_15deg = 'spectra_patches_radius15deg_gal080'
spectra_ref_file     = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080.dat'

l, dltt_ref = np.loadtxt(spectra_ref_file, unpack=1)

# embed()

dls_5deg  = GetDls(spectra_folder_5deg)
dls_10deg = GetDls(spectra_folder_10deg)
dls_15deg = GetDls(spectra_folder_15deg)

fig, ax = subplots(2,3, figsize=(20,12))

# Different lmax plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~
for iLM, LM in enumerate([1200,1900]):
	ax[iLM,0].set_title(r'Radius 5  deg - $\ell_{\rm max}=%d - N=%d$'%(LM,dls_5deg.shape[0]), size=15)
	ax[iLM,1].set_title(r'Radius 10 deg - $\ell_{\rm max}=%d - N=%d$'%(LM,dls_10deg.shape[0]), size=15)
	ax[iLM,2].set_title(r'Radius 15 deg - $\ell_{\rm max}=%d - N=%d$'%(LM,dls_15deg.shape[0]), size=15)
	for lm in [300, 650, 1000]:
		taus_15_CR = GetManyTaus(dls_15deg, dltt_ref, lm, LM, method='CR')
		taus_10_CR = GetManyTaus(dls_10deg, dltt_ref, lm, LM, method='CR')
		taus_5_CR  = GetManyTaus(dls_5deg, dltt_ref, lm, LM, method='CR')

		hist(taus_5_CR,  'knuth', histtype='step', ax=ax[iLM,0], label=r'$\ell_{\rm min} = %d\, \gamma_1=%.2f\pm%.2f$'%(lm,stats.skew(taus_5_CR),err_skew(taus_5_CR)))
		hist(taus_10_CR, 'knuth', histtype='step', ax=ax[iLM,1], label=r'$\ell_{\rm min} = %d\, \gamma_1=%.2f\pm%.2f$'%(lm,stats.skew(taus_10_CR),err_skew(taus_10_CR)))
		hist(taus_15_CR, 'knuth', histtype='step', ax=ax[iLM,2], label=r'$\ell_{\rm min} = %d\, \gamma_1=%.2f\pm%.2f$'%(lm,stats.skew(taus_15_CR),err_skew(taus_15_CR)))
		# hist(taus_5_CR,  'knuth', histtype='step', ax=ax[iLM,0], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,LM,stats.skew(taus_5_CR),err_skew(taus_5_CR)))
		# hist(taus_10_CR, 'knuth', histtype='step', ax=ax[iLM,1], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,LM,stats.skew(taus_10_CR),err_skew(taus_10_CR)))
		# hist(taus_15_CR, 'knuth', histtype='step', ax=ax[iLM,2], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,LM,stats.skew(taus_15_CR),err_skew(taus_15_CR)))

for i in xrange(3):
	ax[0,i].legend(loc='best')
	ax[1,i].legend(loc='best')
	ax[1,i].set_xlabel(r'$\hat{\tau}$', size=15)
	ax[0,i].set_xlim(-0.1,0.1)
	ax[1,i].set_xlim(-0.1,0.1)
	ax[0,i].axvline(0, ls='--', color='grey')
	ax[1,i].axvline(0, ls='--', color='grey')

savefig('plots/taus_smica_halfmission1_cross_halfmission2_filt_200_2000_lmax1200_vs_lmax1900_methodCR.pdf', bboxes_inches='tight')

show()

# fig, ax = subplots(2,3, figsize=(15,5))
# ax[0,0].set_title(r'Radius 5  deg - RK - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[0,1].set_title(r'Radius 10 deg - RK - $N=%d$'%dls_10deg.shape[0], size=15)
# ax[0,2].set_title(r'Radius 15 deg - RK - $N=%d$'%dls_15deg.shape[0], size=15)

# ax[1,0].set_title(r'Radius 5  deg - CR - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[1,1].set_title(r'Radius 10 deg - CR - $N=%d$'%dls_10deg.shape[0], size=15)
# ax[1,2].set_title(r'Radius 15 deg - CR - $N=%d$'%dls_15deg.shape[0], size=15)

# for lm in [300, 650, 1000]:
# 	taus_15    = GetManyTaus(dls_15deg, dltt_ref, lm, lmax)
# 	taus_10    = GetManyTaus(dls_10deg, dltt_ref, lm, lmax)
# 	taus_5     = GetManyTaus(dls_5deg, dltt_ref, lm, lmax)
# 	taus_15_CR = GetManyTaus(dls_15deg, dltt_ref, lm, lmax, method='CR')
# 	taus_10_CR = GetManyTaus(dls_10deg, dltt_ref, lm, lmax, method='CR')
# 	taus_5_CR  = GetManyTaus(dls_5deg, dltt_ref, lm, lmax, method='CR')

# 	taus_5  = taus_5[np.where(taus_5 != np.nan)[0]]
# 	taus_10 = taus_10[np.where(taus_10 != np.nan)[0]]
# 	taus_15 = taus_15[np.where(taus_15 != np.nan)[0]]

# 	embed()

# 	hist(taus_5,  'knuth', histtype='step', ax=ax[0,0], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_5),err_skew(taus_5)))
# 	hist(taus_10, 'knuth', histtype='step', ax=ax[0,1], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_10),err_skew(taus_10)))
# 	hist(taus_15, 'knuth', histtype='step', ax=ax[0,2], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_15),err_skew(taus_15)))

# 	hist(taus_5_CR,  'knuth', histtype='step', ax=ax[1,0], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_5_CR),err_skew(taus_5_CR)))
# 	hist(taus_10_CR, 'knuth', histtype='step', ax=ax[1,1], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_10_CR),err_skew(taus_10_CR)))
# 	hist(taus_15_CR, 'knuth', histtype='step', ax=ax[1,2], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_15_CR),err_skew(taus_15_CR)))

# for i in xrange(3):
# 	ax[0,i].legend(loc='best')
# 	ax[1,i].legend(loc='best')
# 	ax[1,i].set_xlabel(r'$\hat{\tau}$', size=15)
# 	ax[0,i].set_xlim(-0.1,0.1)
# 	ax[1,i].set_xlim(-0.1,0.1)

# show()

# fig, ax = subplots(1,3, figsize=(20,6))
# ax[0].set_title(r'Radius 5  deg - CR - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[1].set_title(r'Radius 10 deg - CR - $N=%d$'%dls_10deg.shape[0], size=15)
# ax[2].set_title(r'Radius 15 deg - CR - $N=%d$'%dls_15deg.shape[0], size=15)

# for lm in [300, 650, 1000]:
# 	taus_15 = GetManyTaus(dls_15deg, dltt_ref, lm, lmax, method='CR')
# 	taus_10 = GetManyTaus(dls_10deg, dltt_ref, lm, lmax, method='CR')
# 	taus_5  = GetManyTaus(dls_5deg, dltt_ref, lm, lmax, method='CR')

# 	hist(taus_5, 'knuth', histtype='step', ax=ax[0], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_5),err_skew(taus_5)))
# 	hist(taus_10, 'knuth', histtype='step', ax=ax[1], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_10),err_skew(taus_10)))
# 	hist(taus_15, 'knuth', histtype='step', ax=ax[2], label=r'$(\ell_{\rm min},\ell_{\rm max}) = (%d,%d)\, \gamma_1=%.2f\pm%.2f$'%(lm,lmax,stats.skew(taus_15),err_skew(taus_15)))

# for i in xrange(3):
# 	ax[i].legend(loc='best')
# 	ax[i].set_xlabel(r'$\hat{\tau}$', size=15)
# 	ax[i].set_xlim(-0.1,0.1)
# 	ax[i].axvline(0, ls='--', color='grey')

# # show()
# savefig('plots/taus_smica_halfmission1_cross_halfmission2_filt_200_2000_lmax'+str(lmax)+'_methodCR.pdf', bboxes_inches='tight')

# close()

# mycosmo = Cosmo({'tau':0.06, 'As':2.1e-9})
# tt = mycosmo.cmb_spectra(2000,dl=1)[:,0]/1e12

# fig, ax = subplots(1,3, figsize=(20,6))
# ax[0].set_title(r'Radius 5  deg - CR - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[1].set_title(r'Radius 10 deg - CR - $N=%d$'%dls_10deg.shape[0], size=15)
# ax[2].set_title(r'Radius 15 deg - CR - $N=%d$'%dls_15deg.shape[0], size=15)

# for i in xrange(dls_5deg.shape[0]):
# 	ax[0].plot(dls_5deg[i,:],color='grey', alpha=0.05)
# ax[0].plot(dltt_ref,'k')
# ax[0].plot(tt,'r')

# for i in xrange(dls_10deg.shape[0]):
# 	ax[1].plot(dls_10deg[i,:],color='grey', alpha=0.05)
# ax[1].plot(dltt_ref,'k')
# ax[1].plot(tt,'r')

# for i in xrange(dls_15deg.shape[0]):
# 	ax[2].plot(dls_15deg[i,:],color='grey', alpha=0.05)
# ax[2].plot(dltt_ref,'k')
# ax[2].plot(tt,'r')


# for i in xrange(3):
# 	# ax[i].legend(loc='best')
# 	ax[i].set_xlabel(r'$\ell$', size=15)
# 	ax[i].set_xlim(2,2000)
# 	ax[i].axhline(0, ls='--', color='grey')

# show()

# embed()










