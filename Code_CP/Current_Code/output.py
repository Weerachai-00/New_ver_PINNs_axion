import scipy.integrate as integrate
import numpy as np
import numpy.random as rd
from fractions import *
import scipy as sp
import sys

###################################################################
###################################################################
####                        Output                             ####
###################################################################
###################################################################

#########################################
#####     Axion Energy Density     ######
#########################################
def axionrho(y,N,n):
    rhoa = np.sum(y[:,2::3][:],axis=-1)
    return rhoa
#########################################
	
#########################################
#####    		 Pressure          ######
#########################################	
def pressure(y,ma_array,N,n):

    #FIX ME!!

    #rho = axionrho(y,N,n)
    #P = np.sum(y[:,1::2][:n]**2,axis=0)-rho

    #P = [np.sum(- 0.5*(ma_array[:n]*y[i,0:-1:2][:n])**2 + 0.5*y[i,1::2][:n]**2) for i in range(N)]
    #P = [np.sum(- 0.5*(ma_array[:n]*y[i,0:-1:3][:n])**2 + 0.5*y[i,1::3][:n]**2) for i in range(N)]
    P = np.sum(ma_array[:]*ma_array[:]*y[:,0:-1:3][:]*y[:,0:-1:3][:] + 0.5*y[:,1::3][:]*y[:,1::3][:],axis=1)
    
    #P=[]	
    #for i in range(N):
    #    p_temp = 0
    #    for j in range(n):
    #        p_temp = p_temp - 0.5*(ma_array[j]*y[i,0:-1:2][j])**2 + 0.5*y[i,1::2][j]**2
    #    P.append(p_temp)

    #print P[0:5], P[-5:]
    #sys.exit()

    return P
#########################################	
	
#########################################
#########        W Axion         ########
#########################################
def w(P,rhoa,N):
    w = P/rhoa
    return w
#########################################
	
#########################################
##  Matter & Radiation energy density  ##
#########################################
def dense(rho_m0,rho_r0,N,y):
	rhom=[]
	rhor=[]
	for i in range(N):
		rhom.append(rho_m0/y[:,-1][i]**3)
		rhor.append(rho_r0/y[:,-1][i]**4)
	return rhom,rhor
#########################################
	
#########################################
########   Total energy density  ########
#########################################
def totalrho(rhom,rhol,rhor,rhoa,N):
    rhom = np.array(rhom)
    rhol = np.array(rhol)
    rhor = np.array(rhor)
    rhoa = np.array(rhoa)
    rhosum = rhom+rhol+rhor+rhoa
    return rhosum
#########################################
	
#########################################
########       Hubble scale      ########
#########################################
def hubble(t,rhosum):	
    H = (1.0/np.sqrt(3.0))*np.sqrt(rhosum[0:len(t)])
    return H
#########################################
	
#########################################
#########        Red shift       ########
#########################################
def redshift(y,N):
    z=[]
    z = 1.0/y[:,-1][0:N] - 1
	#for i in range(N):
	#	z.append(1/y[:,-1][i] - 1)	
    return z	
#########################################
				
###################################################################
####          Dark Matter and Dark Energy Densities            ####
###################################################################	
def dark(rhom,rhor,rhol,rhoa,ma_array,rhosum,n,y):
	ODM=[]
	ODE=[]
	rhoa_de = 0
	rhoa_dm = 0
	for j in range (n):
		if np.sqrt(3)*np.sqrt(rhom[-1]+rhor[-1]+rhol+rhoa[-1])>ma_array[j]:
			rhoa_de = rhoa_de + y[-1,2::3][j]
		else:
			rhoa_dm = rhoa_dm + y[-1,2::3][j]
			
	ODM.append(rhoa_dm/rhosum[-1])
	ODE.append(rhoa_de/rhosum[-1])
	return ODM,ODE
###################################################################
###################################################################
