import numpy as np
dat=np.load('mcmc/posteriorPDF.npy')
med=np.median(dat,axis=0)
hi=np.percentile(dat,100-15.865,axis=0)
lo=np.percentile(dat,15.865,axis=0)
for i in range(len(med)):
	print('%.5f + %.5f - %.5f' %(med[i],hi[i]-med[i],med[i]-lo[i]))

