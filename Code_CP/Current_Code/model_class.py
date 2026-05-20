import numpy as np
import ConfigParser
from sys import argv
import scipy.integrate as integrate
import numpy as np
import numpy.random as rd
from fractions import *
import scipy as sp

config = ConfigParser.RawConfigParser()

class ModelClass(object):
	
	def __init__(self,ifcosmosis=False):
		
		# Things you do when you instantiate the class
		# Read the sys argv
		# Define globally useful things:
		# model number and number of model-specific hyper parameters
		
		# hyper parameter vector set as input for cosmosis
		# read from ini file if not
		
		if ifcosmosis:
			raise Exception('this shit doesnt work')
		else:
			config.read(argv[1])
			self.modnum = config.getint('Model_Selection','Model' )
			self.verbose = config.getboolean('General','verbose')
			if self.modnum == 1:
				self.paramNum=2
			elif self.modnum == 2:
				self.paramNum=4
			elif self.modnum == 3:
				self.paramNum=4
		
	def hyper(self,ifcosmosis=False):
		# hyper parameter vector set as input for cosmosis
		# read from ini file if not
		if ifcosmosis:
			raise Exception('this shit doesnt work')
		else:
			nax=config.getfloat('Hyperparameter','Number of Axions')
			if self.modnum == 1:
				sb = config.getfloat('Hyperparameter','sigma_b')
				c = config.getfloat('Hyperparameter','Dimension')
				hyper=np.vstack((nax,c,sb))
			elif self.modnum == 2:
				kmin = config.getfloat('Hyperparameter','kmin')
				kmax = config.getfloat('Hyperparameter','kmax')
				mmin = config.getfloat('Hyperparameter','mmin')
				mmax = config.getfloat('Hyperparameter','mmax')
				hyper=np.vstack((nax,kmin,kmax,mmin,mmax))
			elif self.modnum == 3:
				b0 = config.getfloat('Hyperparameter','b0')
				sb = config.getfloat('Hyperparameter','sigma_b')
				s1 = config.getfloat('Hyperparameter','s1')
				s2 = config.getfloat('Hyperparameter','s2')
				hyper=np.vstack((nax,s1,s2,b0,sb))
		
		return hyper		
		
#######################################################	
	def diagDoddy(self,hyper):
	
		###################################################################
		###################################################################
		####                        Models                             ####
		###################################################################
		###################################################################

		mo=self.modnum
		n=hyper[0]

		###################################################################
		####              Easther - McAllister Model (1)                ####
		###################################################################

		
		if mo == 1:
			# hyper is (c,sb)
			c=hyper[1]
			sb=hyper[2]
			
			######################################
			####          Kahler, trivial for this model              ####
			######################################		
		
			kk = np.empty((n,n))
			kk.fill(1)
			#kk = np.full((n, 1), 1) 
			kk3=np.diag(kk[:,0])
			kT = kk3.transpose() # transpose of random matrix k
			k2 = np.dot(kk3,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpose of kDr matrix

			######################################
			####            Mass              ####
			######################################
		
		
			L=int(n/c)
			X = sb*(np.random.randn(n, L)) 
			Wc = np.dot(X,(X.T))/L
			mn = reduce(np.dot, [pT,kDrT, Wc, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses from eigenvalues of mn
		
	
		
		####################################################################
		####################################################################


		###################################################################
		####                     LogFlat Model (2)                     ####
		###################################################################

		if mo == 2:
			# hyper is a0,sa,mlow,mup
			kmin=hyper[1]
			kmax=hyper[2]
			mmin=hyper[3]
			mmax=hyper[4]

			######################################
			####          Kahler              ####
			######################################

			k = (np.random.uniform(kmin,kmax,(n,n))) #random matrix k from log normal distribution
			kk = np.exp(-k)
			kT = kk.transpose() # transpose of random matrix k
			k2 = np.dot(kk,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)
			fef2=np.log(fef)
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpos

			######################################
			####            Mass              ####
			######################################

			m = (np.random.uniform(mmin,mmax,(n,n))) #random matrix m from log flat
			mm = np.exp(-m)
			mT = mm.transpose() # transpose of random matrix m
			m2 = np.dot(mm,mT) # symmeterised mass matrix from real axion fields
			mn = reduce(np.dot, [pT,kDrT, m2, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses from eigenvalues of mn

		####################################################################
		####################################################################

		###################################################################
		####                     Bobby's Model (3)                     ####
		###################################################################

		if mo == 3:
			# hyper is s1,s2,b0,sb
			s1=hyper[1]
			s2=hyper[2]
			b0=hyper[3]
			sb=hyper[4]
			# I am setting a0 to 1 here: I think there are implicit units!
			a0=1.

			######################################
			####          Kahler              ####
			######################################

			k = (a0/np.random.uniform(s1,s2,(n,n))) #random matrix k from log normal distribution
			kT = k.transpose() # transpose of random matrix k
			k2 = np.dot(k,kT)  # Construction of symmeterised Kahler matric for real axion fields
			ev,pT = np.linalg.eigh(k2) # calculation of eigen values and eigen vectors
			fef = np.sqrt(ev)
			p = pT.transpose() # tranpose of rotational matrix constructed of eigen vectors
			kD = reduce(np.dot, [p, k2, pT]) #diagonalisation of Kahler metric
			kD[kD < 1*10**-13] = 0 # removal of computational error terms in off diagonal elements
			kDr = np.zeros((n, n))#creation of empty 3x3 matrix
			np.fill_diagonal(kDr, (1/((2**0.5)*np.sqrt(ev))))# matrix for absolving eigen values of kahler metric into axion fields
			#kDr[kDr > 1*10**23] = 0 # remove computational errors in reciprocal matrix
			kDrT = kDr.transpose() # trasnpose of kDr matrix

			######################################
			####            Mass              ####
			######################################

			m = (np.random.uniform(np.log(b0)-sb,np.log(b0)+sb,(n,n))) #random matrix m from log normal distribution
			mm = np.exp(-m)
			mT = mm.transpose() # transpose of random matrix m
			m2 = np.dot(mm,mT) # symmeterised mass matrix from real axion fields
			mn = reduce(np.dot, [pT,kDrT, m2, kDr,p]) # correct mass matrix caclulation
			ma_array,mv = np.linalg.eigh(mn) # reout of masses from eigenvalues of mn

		
		return ma_array,fef

	####################################################################
	####################################################################