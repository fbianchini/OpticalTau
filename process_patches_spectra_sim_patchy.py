import numpy as np
import cPickle as pk
import gzip as gz
from IPython import embed
import glob
from pylab import *
from astropy.visualization import hist
from scipy import stats

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

def GetTau(tt, tt_ref, l, lmin=650, lmax=2000, method='RK'):
	assert (tt.size == tt_ref.size)
	tt_ = tt[np.where((l>=lmin) & (l<=lmax))]
	tt_ref_ = tt_ref[np.where((l>=lmin) & (l<=lmax))]
	if method == 'RK' :
		delta_ell = lmin - lmax
		return np.sum(-0.5 * np.log(tt[lmin:lmax+1]/tt_ref[lmin:lmax+1])) / delta_ell
	elif method == 'CR':
		# return -0.5 * np.log(np.sum(tt[lmin:lmax+1])/np.sum(tt_ref[lmin:lmax+1]))
		return -0.5 * np.log(np.sum(tt_)/np.sum(tt_ref_))

def GetManyTaus(dls, dl_ref, l, lmin=650, lmax=2000, method='RK'):
	assert (dls.shape[1] == dl_ref.shape[0])
	taus = np.zeros(dls.shape[0])
	for i in xrange(dls.shape[0]):
		taus[i] = GetTau(dls[i,:], dl_ref, l, lmin=lmin, lmax=lmax, method=method)
	return taus

def err_skew(taus):
	n = len(taus)
	return np.sqrt(6.*n*(n-1)/((n-2)*(n+1)*(n+3)))

# Params
lmin = 500
lmax = 1200

#Files
spectra_folder_5deg_sim_patchy  = 'spectra_patches_radius5deg_sim_wonoise_patchytau_gal080_MASTER_deltaell20'
spectra_folder_10deg_sim_patchy = 'spectra_patches_radius10deg_sim_wonoise_patchytau_gal080_MASTER_deltaell20'

spectra_folder_5deg_sim  = 'spectra_patches_radius5deg_sim_wonoise_gal080_MASTER_deltaell20'
spectra_folder_10deg_sim = 'spectra_patches_radius10deg_sim_wonoise_gal080_MASTER_deltaell20'

spectra_ref_sim_patchy_file  = 'Dl_sim_wonoise_patchytau_nopixwin_beam_ptsrc_MASTER_delta20.dat'
spectra_ref_sim_file = 'Dl_sim_wonoise_nopixwin_beam_ptsrc_MASTER_delta20.dat'

l, dltt_ref_patchy  = np.loadtxt(spectra_ref_sim_patchy_file, unpack=1)
l, dltt_ref = np.loadtxt(spectra_ref_sim_file, unpack=1)

# fig, ax1 = plt.subplots()
# ax1.plot(l_galonly, dltt_ref_galonly, label='Gal Mask')
# ax1.plot(l_galPS, dltt_ref_galPS, label='Gal + PS Mask')
# ax1.plot(l_smooth, dltt_ref_smooth, label='Gal + PS (smooth) Mask')
# ax1.plot(l, dltt_ref, label=r'Gal + PS Mask MASTER $\Delta\ell=20$')
# ax1.plot(tt, label='Input Theory')
# ax1.set_xlabel(r'$\ell$', size=15)
# ax1.set_ylabel(r'$\mathcal{D}_{\ell}^{TT}[\mu K^2]$', size=15)
# ax1.legend(loc=3)

# axins = zoomed_inset_axes(ax1, 7, loc=2) # zoom-factor: 2.5, location: upper-left
# axins.plot(l_galonly, dltt_ref_galonly, label='Gal Mask')
# axins.plot(l_galPS, dltt_ref_galPS, label='Gal + PS Mask')
# axins.plot(l_smooth, dltt_ref_smooth, label='Gal + PS (smooth) Mask')
# axins.plot(l, dltt_ref, label=r'Gal + PS Mask MASTER $\Delta\ell=20$')
# axins.plot(tt, label='Input Theory')
# x1, x2, y1, y2 = 1500, 1900, 250, 600 # specify the limits
# axins.set_xlim(x1, x2) # apply the x-limits
# axins.set_ylim(y1, y2) # apply the y-limits
# plt.yticks(visible=False)
# plt.xticks(visible=False)
# mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")

# dltt_ref=dltt_ref[2:]


dls_5deg_patchy  = GetDls(spectra_folder_5deg_sim_patchy)
dls_10deg_patchy = GetDls(spectra_folder_10deg_sim_patchy)
dls_5deg  = GetDls(spectra_folder_5deg_sim)
dls_10deg = GetDls(spectra_folder_10deg_sim)

# embed()

# fig, ax = subplots(2,2, figsize=(12,12))
# # ax[0].set_title(r'Radius 5  deg - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[0,0].set_title(r'Radius 5 deg  - CR - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[0,1].set_title(r'Radius 10 deg - CR - $N=%d$'%dls_10deg.shape[0], size=15)
# # ax[0].set_title(r'Radius 15 deg - $N=%d$'%dls_15deg.shape[0], size=15)

fig, ax = subplots(2,3, figsize=(22,10))
fig.suptitle(r'DATA - Radius 5  deg - $N=%d$'%(dls_5deg_patchy.shape[0]), size=15)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min}=%d,\ell_{\rm max}=%d)$'%(lm,LM), size=15)
		taus_patchy  = GetManyTaus(dls_5deg_patchy, dltt_ref_patchy, l, lm, LM, method='CR')
		taus = GetManyTaus(dls_5deg, dltt_ref, l, lm, LM, method='CR')

		# Random patches
		hist(taus_patchy, 'knuth', histtype='step', ax=ax[iLM,ilm], label=r'w/  patchy $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_patchy),err_skew(taus_patchy)))
		hist(taus        ,'knuth', histtype='step', ax=ax[iLM,ilm], label=r' $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus),err_skew(taus)))

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='best')
		ax[i,j].set_xlabel(r'$\hat{\tau}$', size=15)
		ax[i,j].set_xlim(-0.1,0.1)
		ax[i,j].axvline(0, ls='--', color='grey')

savefig('plots/taus_sim_w_wopatchy_wonoise_5deg.pdf', bboxes_inches='tight')

show()

fig, ax = subplots(2,3, figsize=(22,10))
fig.suptitle(r'DATA - Radius 10  deg - $N=%d$'%(dls_10deg_patchy.shape[0]), size=15)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min}=%d,\ell_{\rm max}=%d)$'%(lm,LM), size=15)
		taus_patchy  = GetManyTaus(dls_10deg_patchy, dltt_ref_patchy, l, lm, LM, method='CR')
		taus = GetManyTaus(dls_10deg, dltt_ref, l, lm, LM, method='CR')

		# Random patches
		hist(taus_patchy, 'knuth', histtype='step', ax=ax[iLM,ilm], label=r'w/  patchy $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_patchy),err_skew(taus_patchy)))
		hist(taus        ,'knuth', histtype='step', ax=ax[iLM,ilm], label=r' $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus),err_skew(taus)))

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='best')
		ax[i,j].set_xlabel(r'$\hat{\tau}$', size=15)
		ax[i,j].set_xlim(-0.1,0.1)
		ax[i,j].axvline(0, ls='--', color='grey')

savefig('plots/taus_sim_w_wopatchy_wonoise_10deg.pdf', bboxes_inches='tight')

# embed()










