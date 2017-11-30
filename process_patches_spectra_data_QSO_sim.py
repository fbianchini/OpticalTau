import numpy as np
import cPickle as pk
import gzip as gz
from IPython import embed
import glob
from pylab import *
from astropy.visualization import hist
from scipy import stats
import seaborn as sns
# sns.set(rc={"figure.figsize": (8, 6)})

def SetPlotStyle():
	rc('text',usetex=True)
	rc('font',**{'family':'serif','serif':['Computer Modern']})
	plt.rcParams['axes.linewidth']  = 3.
	plt.rcParams['axes.labelsize']  = 28
	plt.rcParams['axes.titlesize']  = 22
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 18
	plt.rcParams['xtick.major.size'] = 7
	plt.rcParams['ytick.major.size'] = 7
	plt.rcParams['xtick.minor.size'] = 3
	plt.rcParams['ytick.minor.size'] = 3
	plt.rcParams['legend.fontsize']  = 20
	plt.rcParams['legend.frameon']  = False

	plt.rcParams['xtick.major.width'] = 1
	plt.rcParams['ytick.major.width'] = 1
	plt.rcParams['xtick.minor.width'] = 1
	plt.rcParams['ytick.minor.width'] = 1

	plt.clf()
	sns.set_style("ticks", {'figure.facecolor': 'grey'})

SetPlotStyle()

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
spectra_folder_5deg_data  = 'spectra_patches_radius5deg_gal080_MASTER_deltaell20'
spectra_folder_10deg_data = 'spectra_patches_radius10deg_gal080_MASTER_deltaell20'

spectra_folder_5deg_sim  = 'spectra_patches_radius5deg_sim_wnoise_gal080_MASTER_deltaell20'
spectra_folder_10deg_sim = 'spectra_patches_radius10deg_sim_wnoise_gal080_MASTER_deltaell20'

spectra_ref_sim_file   = 'Dl_sim_wnoise_nopixwin_beam_ptsrc_MASTER_delta20.dat'
spectra_ref_data_file  = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_MASTER_deltaell20.dat'
spectra_QSO_5deg_file  = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_QSOvoid_radius5.0deg_MASTER_deltaell20.dat'
spectra_QSO_10deg_file = 'Dl_smica_halfmission1_cross_halfmission2_filt_200_2000_gal080_QSOvoid_radius10.0deg_MASTER_deltaell20.dat'

l, dltt_ref_sim   = np.loadtxt(spectra_ref_sim_file, unpack=1)
l, dltt_ref_data  = np.loadtxt(spectra_ref_data_file, unpack=1)
l, dltt_QSO_5deg  = np.loadtxt(spectra_QSO_5deg_file, unpack=1)
l, dltt_QSO_10deg = np.loadtxt(spectra_QSO_10deg_file, unpack=1)

# l_smooth, dltt_ref_smooth = np.loadtxt('Dl_sim_wonoise_nopixwin_beam_ptsrc_ptsrc_smooth.dat', unpack=1)
# l_galonly, dltt_ref_galonly = np.loadtxt('Dl_sim_wonoise_nopixwin_nobeam_nofilt_noptsrc.dat', unpack=1)
# l_galPS, dltt_ref_galPS = np.loadtxt('Dl_sim_wonoise_nopixwin.dat', unpack=1)


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

# embed()

dls_5deg_sim   = GetDls(spectra_folder_5deg_sim)
dls_10deg_sim  = GetDls(spectra_folder_10deg_sim)
dls_5deg_data  = GetDls(spectra_folder_5deg_data)
dls_10deg_data = GetDls(spectra_folder_10deg_data)

# fig, ax = subplots(2,2, figsize=(12,12))
# # ax[0].set_title(r'Radius 5  deg - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[0,0].set_title(r'Radius 5 deg  - CR - $N=%d$'%dls_5deg.shape[0], size=15)
# ax[0,1].set_title(r'Radius 10 deg - CR - $N=%d$'%dls_10deg.shape[0], size=15)
# # ax[0].set_title(r'Radius 15 deg - $N=%d$'%dls_15deg.shape[0], size=15)

# Plot paper
fig, ax = plt.subplots()# figsize=(22,12))
plt.title(r'Radius 10  deg - $N=%d$'%(dls_10deg_data.shape[0]), size=20)# y=1.08)
taus_data = GetManyTaus(dls_10deg_data, dltt_ref_data, l, 300, 1900, method='CR')
taus_sim  = GetManyTaus(dls_10deg_sim, dltt_ref_sim, l, 300, 1900, method='CR')

# Random patches
sns.distplot(taus_data, ax=ax, norm_hist=True, hist_kws={'histtype':'step', 'linewidth':2}, label='Data')
sns.distplot(taus_sim,  ax=ax, norm_hist=True, hist_kws={'histtype':'step', 'linewidth':2}, label='Sims')
ax.axvline(GetTau(dltt_QSO_10deg, dltt_ref_data, l, lmin=300, lmax=1900, method='CR'), label='QSO patch', color='tomato')
ax.legend(loc='upper left')
ax.set_xlabel(r'$\delta\tau$')#, size=15)
ax.set_xlim(-0.05,0.05)
ax.axvline(0, ls='--', color='grey')
# plt.subplots_adjust(top=0.9)
plt.tight_layout() # Or equivalently,  "plt.tight_layout()"
#
savefig('plots/hits_plot.pdf', bboxes_inches='tight')
# savefig('plots/taus_data_sim_wnoise_QSOpatch_10deg_paper.pdf', bboxes_inches='tight')

exit()


fig, ax = subplots(2,3, figsize=(22,12))
fig.suptitle(r'Radius 5  deg - $N=%d$'%(dls_5deg_data.shape[0]), size=20)#, y=1.08)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min},\ell_{\rm max})=(%d,%d)$'%(lm,LM), size=20, )
		taus_data = GetManyTaus(dls_5deg_data, dltt_ref_data, l, lm, LM, method='CR')
		taus_sim  = GetManyTaus(dls_5deg_sim, dltt_ref_sim, l, lm, LM, method='CR')

		# Random patches
		# hist(taus_data, 'knuth', histtype='step', ax=ax[iLM,ilm], label=r'DATA $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_data),err_skew(taus_data)))
		# hist(taus_sim,  'knuth', histtype='step', ax=ax[iLM,ilm], label=r'SIMS $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_sim),err_skew(taus_sim)))
		sns.distplot(taus_data, ax=ax[iLM,ilm], norm_hist=True, hist_kws={'histtype':'step', 'linewidth':3}, label='Data')
		sns.distplot(taus_sim,  ax=ax[iLM,ilm], norm_hist=True, hist_kws={'histtype':'step', 'linewidth':3}, label='Sims')


		# QSO patch
		ax[iLM,ilm].axvline(GetTau(dltt_QSO_5deg, dltt_ref_data, l, lmin=lm, lmax=LM, method='CR'), label='QSO patch', color='tomato')

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='upper left')
		ax[1  ,j].set_xlabel(r'$\hat{\tau}$')#, size=15)
		ax[i,j].set_xlim(-0.1,0.1)
		ax[i,j].axvline(0, ls='--', color='grey')

plt.subplots_adjust(top=0.9)
# plt.tight_layout() # Or equivalently,  "plt.tight_layout()"

savefig('plots/taus_data_sim_wnoise_QSOpatch_5deg_new.pdf', bboxes_inches='tight')

# show()

# embed()

fig, ax = subplots(2,3, figsize=(22,12))
fig.suptitle(r'Radius 10  deg - $N=%d$'%(dls_10deg_data.shape[0]), size=20)# y=1.08)

for iLM, LM in enumerate([1200,1900]):
	for ilm, lm in enumerate([300, 650, 1000]):
		ax[iLM,ilm].set_title(r'$(\ell_{\rm min},\ell_{\rm max})=(%d,%d)$'%(lm,LM), size=20,)
		taus_data = GetManyTaus(dls_10deg_data, dltt_ref_data, l, lm, LM, method='CR')
		taus_sim  = GetManyTaus(dls_10deg_sim, dltt_ref_sim, l, lm, LM, method='CR')

		# Random patches
		# hist(taus_data, 'knuth', histtype='step', ax=ax[iLM,ilm], label=r'DATA $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_data),err_skew(taus_data)))
		# hist(taus_sim,  'knuth', histtype='step', ax=ax[iLM,ilm], label=r'SIMS $\gamma_1=%.2f\pm%.2f$'%(stats.skew(taus_sim),err_skew(taus_sim)))
		sns.distplot(taus_data, ax=ax[iLM,ilm], norm_hist=True, hist_kws={'histtype':'step', 'linewidth':3}, label='Data')
		sns.distplot(taus_sim,  ax=ax[iLM,ilm], norm_hist=True, hist_kws={'histtype':'step', 'linewidth':3}, label='Sims')


		# QSO patch
		ax[iLM,ilm].axvline(GetTau(dltt_QSO_10deg, dltt_ref_data, l, lmin=lm, lmax=LM, method='CR'), label='QSO patch', color='tomato')

for i in xrange(2):
	for j in xrange(3):
		ax[i,j].legend(loc='upper left')
		ax[1,j].set_xlabel(r'$\hat{\tau}$')#, size=15)
		ax[i,j].set_xlim(-0.05, 0.05)
		ax[i,j].axvline(0, ls='--', color='grey')

plt.subplots_adjust(top=0.9)#, hspace=0.2 )
# plt.tight_layout() # Or equivalently,  "plt.tight_layout()"

savefig('plots/taus_data_sim_wnoise_QSOpatch_10deg_new.pdf')#, bboxes_inches='tight')










