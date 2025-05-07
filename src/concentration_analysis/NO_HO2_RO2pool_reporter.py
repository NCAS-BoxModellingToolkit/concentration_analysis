'''code to return the concentrations of: HO2, NO and RO2pool'''
# concentration at
# end of simulation is found and saved to file

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import os

# user-defined variables start --------------------------------

# set path to results
res_path = str('/Users/user/Documents/GitHub/PyCHAM/' +
	'PyCHAM/output/ex_chem_scheme/flow_reactor_example_run_output')

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# path to save to
save_path = str('/Users/user/Library/CloudStorage/OneDrive' +
	'-TheUniversityofManchester/SOAPRA/Lukas_Share')

file_name = str('NO_HO2_RO2pool_test.txt')


# user-defined variables end --------------------------------

# define function
def NOHO2RO2pool_reporter(res_path, PyCHAM_path, save_path, file_name):

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

	# names of components
	comp_names = self.ro_obj.names_of_comp

	# indices of components in the RO2 pool
	RO2poolindx = self.ro_obj.gi['RO2pooli']

	# convert gas-phase concentrations from ppb to molecules/cm^3
	yrec[:, 0:nc] = yrec[:, 0:nc]*cfac

	# prepare to hold concentration of each component at end of simulation
	NOHO2RO2pool = np.zeros((1, 3))

	# store gas-phase concentration at end of simulation (molecules/cm^3)
	if ('NO' in comp_names):
		NOHO2RO2pool[0, 0] = yrec[-1, comp_names.index('NO')]
	if ('HO2' in comp_names):
		# store gas-phase concentration at end of simulation (molecules/cm^3)
		NOHO2RO2pool[0, 1] = yrec[-1, comp_names.index('HO2')]
	# store gas-phase concentration at end of simulation (molecules/cm^3)
	NOHO2RO2pool[0, 2] = np.sum(yrec[-1, RO2poolindx])

	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))

	np.savetxt(str(save_path + '/' + file_name), NOHO2RO2pool, delimiter='	')
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
NOHO2RO2pool_reporter(res_path, PyCHAM_path, save_path, file_name)