import numpy as np
from fractions import *
import scipy as sp

###################################################################
####                    Initial rho array                      ####
###################################################################

def rhoinitial(phidotin_array,phiin_array,ma_array,n):
	rhoin_array=[]
	for i in range(n):	
		rhoin_array.append( 0.5*phidotin_array[i]**2 + 0.5*(ma_array[i]**2)*phiin_array[i]**2 )
	return rhoin_array
	
###################################################################	

###################################################################
####                    Initial y array                        ####
###################################################################
	
def yinitial(n,phiin_array,phidotin_array,rhoin_array,ain):
	y0=[]
	for i in range(n):
		y0.append(phiin_array[i])
		y0.append(phidotin_array[i])
		y0.append(rhoin_array[i])
	y0.append(ain)
	return y0	
	
###################################################################	

###################################################################
####         Equations of Motion Function (Phi and Rho)        ####
###################################################################

def deriv_wfromphi(y,t,n,n_cross,ma_array,rho_m0,rho_r0,rhol):
        crossing_index = [0]*n
	func=[]
	####### Sum rho axions first (sum from y[2] + y[5] + ... + y[3n-1])
	rho_ax=sum(y[2::3])
	#rho_ax = 0
	#for i in range (n):
		#rho_ax = rho_ax + 0.5*ma_array[i]**2*y[3*i]**2+0.5*y[3*i+1]**2 
	####### Sum rho axions from phi and phidot (Do we need this? which one should we use?)
	#rho_ax_f = 0
	#for i in range(n):
	#	rho_ax_f = rho_ax_f + 0.5*ma_array[i]*y[3*i]*y[3*i] + 0.5*y[3*i+1]*y[3*i+1]
	####### Start filling equations
	####### To check these conditions, we need crossing_index as a global parameter which will be updated every time the condition is met.

	for i in range(n):
		###### update crossing_index
		###### The logic is following:
		###### We start counting when w crosses from negative to positive and crossing index is even
		###### After that crossing index is now odd and we count again when w crosses from positive to negative
		###### After that crossing index is now even and we count again when w crosses from negative to positive
		###### We carry through three equations of motion in order to keep the structure of the code the same throughout. Allowing the field to osciallate for the defined number of crossings for the equation of state we then disregard phi_dot and set this to zero. The field then behaves matter without the computationally limiting friction term 
		######
		
		if ( (crossing_index[i] % 2 == 0 and ma_array[i]*y[3*i]*y[3*i] < y[3*i+1]*y[3*i+1] and y[3*i+1] > 0) or (crossing_index[i] % 2 == 1 and ma_array[i]*y[3*i]*y[3*i] > y[3*i+1]*y[3*i+1] and y[3*i+1] < 0) ):
			crossing_index[i] += 1
		####### phi dot part
		func.append(y[3*i+1])
		#if crossing_index[i] < n_cross:
			#func.append(y[3*i+1])
		#else: ### After cross n_cross times, phidot = 0
			#func.append(0)
		####### phi dot part
		if crossing_index[i] < n_cross :
			func.append( -np.sqrt(3)*(np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol))*y[3*i+1] - (ma_array[i]**2)*y[3*i] )
		else: ### After cross n_cross times, the mass term is switched off in order to drag phi down to zero (or simply put phiddot = 0)
			func.append( -np.sqrt(3)*(np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol))*y[3*i+1] )
			#func.append(0)
		####### rho dot part
		if crossing_index[i] < n_cross  :
			func.append( -3/np.sqrt(3)*np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol)*y[3*i+1]*y[3*i+1] ) ### rho + p = phidot^2
		else: ### After cross n_cross times, P = 0 and rho = 1/a^3
			func.append( -3/np.sqrt(3)*np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol)*y[3*i+2] )	
	func.append((1/np.sqrt(3)*np.sqrt(rho_ax*y[-1]**2 + rho_m0/y[-1] + rho_r0/y[-1]**2 + rhol*y[-1]**2)))
	print crossing_index 

	return func


########################################################################################################################################
########################################################################################################################################

###################################################################
####             Equations of Motion Function (Rho)            ####
###################################################################

def deriv_rho(y,t,n,ma_array,rho_m0,rho_r0,rhol): 
	func=[]
	rho_ax=sum(y[:-1])
	for i in range(n):
		if np.sqrt(3)*np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol) > 2*(ma_array[i]):
			func.append((-3/np.sqrt(3)*np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol)*(1-1)*(y[i])))
		else:
			func.append((-3/np.sqrt(3)*np.sqrt(rho_ax + rho_m0/y[-1]**3 + rho_r0/y[-1]**4 + rhol)*(y[i])))	
	func.append((1/np.sqrt(3)*np.sqrt(rho_ax*y[-1]**2 + rho_m0/y[-1] + rho_r0/y[-1]**2 + rhol*y[-1]**2)))
	return func

###################################################################
