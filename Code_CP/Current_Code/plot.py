import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["axes.facecolor"] = 'w'
plt.rcParams["savefig.facecolor"] = 'w'

def cosmology(rhoa,rhosum,rhor,rhom,y):
	plt.subplot(1,1,1)
	plt.title('N- Axion Cosmology')
	plt.plot(y[:,-1], rhoa,'blue',label=r'$\rho_a$')
	plt.plot(y[:,-1], rhosum,'green',label=r'$\rho_{total}$')
	plt.plot(y[:,-1], rhor,'orange',label=r'$\rho_r$')
	plt.plot(y[:,-1], rhom,'red',label=r'$\rho_m$')
	plt.axvline(0.000333222, color='k', linestyle='--',label=r'$z_{eq}$')
	plt.legend(loc='upper right')
	plt.ylabel(r'$\rho$ (GeV)$^4$', fontsize=16)
	plt.xlabel(r'$a$', fontsize=16)
	plt.grid(True)
	plt.xscale('log')
	plt.yscale('log')
	plt.show()

def massspec(mo,lma_array):
	plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lma_array, 100,facecolor='orange', normed=True,label='$Mass$')
	plt.legend(loc='upper left')
	plt.ylabel('Probability')
	plt.xlabel(r'Log Eigenvalues')
	plt.title(r'Mass Spectrum for Model {0}'. format(mo))
	plt.show()		


def fspec(mo,lfef):		
	plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lfef, 50,facecolor='red', normed=True,label='$f_{eff}$')
	plt.legend(loc='upper left')
	plt.ylabel('Probability')
	plt.xlabel(r'Log Eigenvalues')
	plt.title(r'$f$ Spectrum for Model {0}'. format(mo))
	plt.show()	
	
def mfspec(mo,lfef,lma_array):
	plt.subplot(1,1,1)
	n, bins, patches = plt.hist(lma_array, 100,facecolor='orange', normed=True,label='$Mass$')
	n, bins, patches = plt.hist(lfef, 50,facecolor='red', normed=True,label='$f_{eff}$')
	plt.legend(loc='upper left')
	plt.ylabel('Probability')
	plt.xlabel(r'Log Eigenvalues')
	plt.title(r'Mass and $f$ Spectrum for Model {0}'. format(mo))
	plt.show()	