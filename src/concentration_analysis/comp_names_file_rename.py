'''code to duplicate and rename the file containing component names'''
# names of components are stored under a user-defined file name

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

file_name = str('model_names.txt')


# user-defined variables end --------------------------------

# define function
def comp_names_rename(res_path, PyCHAM_path, save_path, file_name):

	# ensure PyCHAM can be seen
	sys.path.append(PyCHAM_path)
	import retr_out

	# create the self object so that results path is stored
	self = self_def(res_path)

	# import results
	for prog in retr_out.retr_out(self):
		prog = prog

	# names of components
	comp_names = np.array(self.ro_obj.names_of_comp, dtype=str)

	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))
	
	np.savetxt(str(save_path + '/' + file_name), comp_names, fmt='%s', 
		delimiter='	')
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
comp_names_rename(res_path, PyCHAM_path, save_path, file_name)