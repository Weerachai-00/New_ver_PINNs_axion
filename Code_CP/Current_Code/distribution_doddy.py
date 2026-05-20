print 'importing...'

import numpy as np
import model_class
import ConfigParser
from sys import argv


config = ConfigParser.RawConfigParser()
config.read(argv[1])
fileout = config.get('General','Distribution Outputs')


print 'reading ini'

x=model_class.ModelClass()
hyper=x.hyper()

print 'doing diagonilastion'

ma_array, fef = x.diagDoddy(hyper)

lma_array=np.log(ma_array)
lfef=np.log(fef)

if x.verbose==True:
	print "f_effective", "Mass"
	for i in range (0,len(ma_array)):
		print fef[i] , ma_array[i]

outs = np.vstack((fef,ma_array))
print 'saving outputs to file', fileout
np.savetxt(fileout, outs.T)


#if sys.argv[2]=="mspec":
#	plot.massspec(mo, lma_array)
	
#if sys.argv[2]=="fspec":
#	plot.fspec(mo, lfef)
	
#if sys.argv[2]=="mfspec":
#	plot.mfspec(mo, lfef, lma_array)		
