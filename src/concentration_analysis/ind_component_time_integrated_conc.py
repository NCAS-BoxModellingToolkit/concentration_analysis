'''code to print time-integrated simulated concentrations of individual components'''
# concentrations may be for any combination of phases

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import matplotlib.pyplot as plt
import os

# user-defined variables start --------------------------------

# base path to use
base_path = '/Users/user/Library/CloudStorage/OneDrive-TheUniversityofManchester/'

# set path to results
res_path = str(base_path + 'NCAS/MAN_OFR/PyCHAM_outputs/Aircraft/test_experiment_31_07_2025')

# path to PyCHAM
PyCHAM_path = str(base_path + 'GitHub/PyCHAM/PyCHAM')

# chemical scheme name(s) of component to plot
plot_name = ['OH']
# phase(s) to plot, e.g. ['g', 'p', 'w'] for gas, particle and surface
phase = ['g']

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, plot_name, phase, PyCHAM_path):

	# create the self object so that results path is stored
	self = self_def(res_path)

	# ensure PyCHAM can be seeb
	sys.path.append(PyCHAM_path)
	import retr_out

	# import results
	for prog in retr_out.retr_out(self):
		prog = prog

	# get concentrations
	yrec = np.zeros((self.ro_obj.yrec.shape[0], 
		self.ro_obj.yrec.shape[1]))
	yrec[:, :] = self.ro_obj.yrec[:, :]

	# get molar masses of component (g/mol)
	y_MM = self.ro_obj.comp_MW

	# get time (hours) through simulation
	thr = self.ro_obj.thr

	# convert time to s
	time_s = thr*3600.

	# get number of components
	nc = self.ro_obj.nc

	# get names of components
	comp_names = self.ro_obj.names_of_comp

	# loop through components
	for i in range(len(plot_name)):

		# get index of this component
		ci = comp_names.index(plot_name[i])

		# get number of non-particle surfaces
		ns = self.ro_obj.wf	

		if 'g' in phase: # gas-phase concentrations
			yg = yrec[:, ci] # (ppb)

			# factor to convert ppb to molecules/cm^3
			cfac = np.array((self.ro_obj.cfac)).reshape(-1)
		
			# convert to molecules/cm^3
			yg = yg*cfac
		
			# integrate over time (molecules.s/cm^3)
			ygint = sum(((yg[0:-1]+yg[1::])/2.)*(time_s[1::]-time_s[0:-1]))
			print(str(plot_name[i] + ' integrated over time: ' + 
				"{:e}".format(ygint) + ' molecules.s/cm^3'))

		if 'p' in phase: # particle-phase concentrations

			# number of particle size bins
			nsb = self.ro_obj.nsb-ns
			
			# molecules/cm^3
			yp = yrec[:, nc:nc*(nsb+1)]

			# sum concentrations for this component over bins
			yp = np.sum(yp[:, ci::nc], axis=1)

			# convert to ug/m^3
			yp = ((yp*1.e6/si.N_A)*y_MM[ci]*1.e6)

		if 'w' in phase: # non-particle surface concentrations

			# number of particle size bins
			nsb = self.ro_obj.nsb-ns
		
			# molecules/cm^3
			yw = yrec[:, nc*(nsb+1)::]

			# sum concentrations for this component over bins
			yw = np.sum(yw[:, ci::nc], axis=1)

			# convert to ug/m^3
			yw = ((yw*1.e6/si.N_A)*y_MM[ci]*1.e6)
	
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
conc_plot(res_path, plot_name, phase, PyCHAM_path)