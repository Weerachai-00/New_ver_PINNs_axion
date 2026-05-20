"""
Quick histogram script to look at derived parameters
Doddy, Feb 2016
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# derived params are [log10(M200,Mcore), logarithmic slope]
samples = np.loadtxt('test_dist.dat',unpack=True)


#################################
# Plotting
###############################

F1 = 20 # Axes font size
F2 = 12 # Legend font size


# Setting the font structure

rc = mpl.rcParams # Font structure is called rc now
rc['text.usetex'] = True # Tex fonts
rc['font.family'] = 'serif'
rc['font.serif'].insert(0,'cm') # Default font is computer modern for latex
rc['font.size'] = F1
rc['xtick.labelsize'] = 'small'
rc['ytick.labelsize'] = 'small'
rc['legend.fontsize'] = F2

nbins = 100 


mass = samples[1,:]
feff = samples[0,:]

mass=np.log10(mass)
feff=np.log10(feff)

fig,ax=plt.subplots()
n, bins, patches = plt.hist(mass,nbins,normed=1,histtype='stepfilled') #  create histogram
ax.set_xlabel(r'$\log_{10}m_a$',fontsize=F1)
fig.savefig("test_mass_hist.png")

fig,ax=plt.subplots()
n, bins, patches = plt.hist(feff,nbins,normed=1,histtype='stepfilled') #  create histogram
ax.set_xlabel(r'$\log_{10}f_{\rm eff}$',fontsize=F1)
fig.savefig("test_feff_hist.png")


