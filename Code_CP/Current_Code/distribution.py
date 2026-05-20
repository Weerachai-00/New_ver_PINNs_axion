import numpy as np
import diagonalisation
import plot
import ConfigParser
import sys
from sys import argv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import rc


config = ConfigParser.RawConfigParser()
#config.read("configuration_card.ini")
config.read(argv[1])

n = config.getint('General','Number of Axions')
mo = config.getint('Model_Selection','Model' )
c = config.getfloat('Model_Selection','Dimension')
a0 = config.getfloat('Hyperparameter','a0')
sa = config.getfloat('Hyperparameter','sigma_a')
b0 = config.getfloat('Hyperparameter','b0')
sb = config.getfloat('Hyperparameter','sigma_b')
s1 = config.getfloat('Hyperparameter','s1')
s2 = config.getfloat('Hyperparameter','s2')

phi_range = config.getfloat('Initial Conditions','phi_in_range')
phidotin = config.getfloat('Initial Conditions','phidot_in')

ma_array, fef, phiin_array, phidotin_array = diagonalisation.diag(mo,c,n,a0,b0,sa,sb,s1,s2,phi_range,phidotin)

lma_array=np.log(ma_array)
lfef=np.log(fef)


print "f_effective", "Mass"
for i in range (0,len(ma_array)):
	print fef[i] , ma_array[i]

if sys.argv[2]=="mspec":
	plot.massspec(mo, lma_array)
	
if sys.argv[2]=="fspec":
	plot.fspec(mo, lfef)
	
if sys.argv[2]=="mfspec":
	plot.mfspec(mo, lfef, lma_array)		
