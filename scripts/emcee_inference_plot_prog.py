import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-p", "--post-loc", dest="post_loc", help="location of posterior samples directory")
parser.add_option("-n", "--ndim", dest="ndim", help="number of dimensions")
(options, args) = parser.parse_args()
post_loc = options.post_loc
ndim = int(options.ndim)

if ndim == 11:
	data = np.loadtxt(post_loc + '/chain_incremental.dat', unpack=True, usecols=(1,2,3,4,5,6,7,8,9,10,11))
	labels = ['mc', 'q', 'mc1', 'q1', 'dL', 'i', 't0', 'phi0', 'ra', 'sin(dec)', 'pol']
else: 
	data = np.loadtxt(post_loc + '/chain_incremental.dat', unpack=True, usecols=(1,2,3,4,5,6,7,8,9))
	labels = ['mc', 'q', 'dL', 'i', 't0', 'phi0', 'ra', 'sin(dec)', 'pol']

samples = np.swapaxes(data, 0,1)

nwalkers = 100
num_iter = int(np.shape(data)[1]/nwalkers)

data = np.reshape(samples, (num_iter, nwalkers, ndim))
data = np.swapaxes(data, 0,1)

plt.figure(figsize=(10,10))
for idx in range(ndim):
	plt.subplot2grid((ndim,6), (idx,0), colspan=5)
	plt.plot(data[:,:,idx].T, color='k', alpha=0.1, lw=0.5)
	plt.ylabel(labels[idx])
	plt.subplot2grid((ndim,6), (idx,5), colspan=1)
	plt.hist(samples[:,idx], histtype='step', bins=50, orientation='horizontal')
plt.savefig(post_loc + '/plot_incremental_progress.png', dpi=300)
