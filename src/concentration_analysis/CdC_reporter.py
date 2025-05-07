'''code to return the concentration and rate (against time) of concentration change'''
# concentration and rate (against time) of concentration change per component at
# end of simulation is found and saved to file

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import os

# user-defined variables start --------------------------------

# set path to results
res_path = str('/Users/user/Documents/GitHub/PyCHAM/' +
	'PyCHAM/output/ex_chem_scheme/Flow_Reactor_Example_Run_Output')

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# path to save to
save_path = str('/Users/user/Library/CloudStorage/OneDrive' +
	'-TheUniversityofManchester/SOAPRA/Lukas_Share')

file_name = str('CdC_test.txt')


# user-defined variables end --------------------------------

# define function
def CdC_reporter(res_path, PyCHAM_path, save_path, file_name):

	# ensure PyCHAM can be seen
	sys.path.append(PyCHAM_path)
	import retr_out

	# create the self object so that results path is stored
	self = self_def(res_path)

	# import results
	for prog in retr_out.retr_out(self):
		prog = prog

	# get the concentrations of components (columns) with time (rows)
	yrec = np.zeros((self.ro_obj.yrec.shape[0], 
		self.ro_obj.yrec.shape[1]))
	yrec[:, :] = self.ro_obj.yrec[:, :]

	# get the conversion factor for ppb to molecules/cm^3
	cfac = np.array((self.ro_obj.cfac)).reshape(-1, 1)

	# number of components
	nc = int((self.ro_obj.nc))

	# times
	thr = np.array((self.ro_obj.thr)).reshape(-1, 1)

	# convert gas-phase concentrations from ppb to molecules/cm^3
	yrec[:, 0:nc] = yrec[:, 0:nc]*cfac

	# prepare to hold concentration and rate of concentration 
	# change of each component at end of simulation
	cdc = np.zeros((nc, 2))

	# store gas-phase concentration at end of simulation (molecules/cm^3)
	cdc[:, 0] = yrec[-1, 0:nc]

	# store rate of change (wrt time) at end of simulation (molecules/cm^3/s)
	cdc[:, 1] = (yrec[-1, 0:nc]-yrec[-2, 0:nc])/((thr[-1]-thr[-2])*3600.)
	
	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))

	np.savetxt(str(save_path + '/' + file_name), cdc, delimiter='	')

		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
CdC_reporter(res_path, PyCHAM_path, save_path, file_name)