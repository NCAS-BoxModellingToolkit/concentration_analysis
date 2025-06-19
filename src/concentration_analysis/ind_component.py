'''code to plot simulated concentrations of individual components'''
# concentrations may be for any combination of phases

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import matplotlib.pyplot as plt
import os

# user-defined variables start --------------------------------

# set path to results
res_path = str('/Users/user/Library/CloudStorage/' +
	'OneDrive-TheUniversityofManchester/NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-3w_mt2p40')

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# chemical scheme name(s) of component to plot
plot_name = ['DNOMCATECHOL', 'NOMCATECHOL', 'DNGUAIACOL', 'NGUAIACOL', 'OMPBZQONE', 'GUAIAOXMUC', 'NCATECHOL', 'OMCATECHOL', 'OMC4CO142OH', 'OMC5CO14OH', 'OMCATPBZQONE']

# phase(s) to plot
phase = ['g', 'p', 'w']

# path to observations
csv_path = str('/Users/user/Library/CloudStorage/OneDrive-TheUniversity' +
	'ofManchester/NCAS/MCM_working_group/guaiacol/guaiacol_gas_phase_obs.csv')

# column of observations file containing times
t_col_indx = 0
# column of observations file containing concentrations
m_col_indx = 1

# state which components in plot_name have corresponding observations
obs_plot = np.zeros((len(plot_name)))
obs_plot[0] = 0

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, plot_name, phase, PyCHAM_path, csv_path, obs_plot, 
	t_col_indx, m_col_indx):

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

		# prepare plots
		fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

		if 'g' in phase: # gas-phase concentrations
			yg = yrec[:, ci] # (ppb)

			# factor to convert ppb to molecules/cm^3
			cfac = np.array((self.ro_obj.cfac)).reshape(-1)
		
			# convert to molecules/cm^3
			yg = yg*cfac
		
			# convert to ug/m^3
			yg = ((yg*1.e6/si.N_A)*y_MM[ci]*1.e6)

			# plot against time (hours)
			ax0.plot(thr, yg, label = str('simulated ' + plot_name[i] + 
				' gas-phase'))

			if (obs_plot[i] == 1):
				# open observed concentration
				wb = np.loadtxt(csv_path, delimiter = ',', 
					skiprows = 1, dtype='str')
				# get observed time through experiment and 
				# particle mass concentration
				obs_thr = wb[:, t_col_indx].astype('float')
				obs_g = wb[:, m_col_indx].astype('float')
				ax0.plot(obs_thr, obs_g, 'k', label = str('observed ' + 
					plot_name[i] + ' gas-phase'))

		if 'p' in phase: # particle-phase concentrations

			# number of particle size bins
			nsb = self.ro_obj.nsb-ns
			
			# molecules/cm^3
			yp = yrec[:, nc:nc*(nsb+1)]

			# sum concentrations for this component over bins
			yp = np.sum(yp[:, ci::nc], axis=1)

			# convert to ug/m^3
			yp = ((yp*1.e6/si.N_A)*y_MM[ci]*1.e6)

			# plot against time (hours)
			ax0.plot(thr, yp, label = str(plot_name[i] + ' particle-phase'))

		if 'w' in phase: # non-particle surface concentrations

			# number of particle size bins
			nsb = self.ro_obj.nsb-ns
		
			# molecules/cm^3
			yw = yrec[:, nc*(nsb+1)::]

			# sum concentrations for this component over bins
			yw = np.sum(yw[:, ci::nc], axis=1)

			# convert to ug/m^3
			yw = ((yw*1.e6/si.N_A)*y_MM[ci]*1.e6)

			# plot against time (hours)
			ax0.plot(thr, yw, label = str(plot_name[i] + ' wall-phase'))
	

		ax0.set_ylabel(str('mass concentration ('  + 
			'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)'), fontsize = 14)
		ax0.set_xlabel(str('time (hours)'), fontsize = 14)
		ax0.yaxis.set_tick_params(labelsize = 14, 
			direction = 'in', which='both')
		ax0.xaxis.set_tick_params(labelsize = 14, 
			direction = 'in', which='both')
		ax0.legend()
		plt.tight_layout()
		# make directory if not already existing
		if (os.path.isdir(str(res_path + '/images')) == False):
			os.mkdir(str(res_path + '/images'))
		plt.savefig(str(res_path + '/images/' + plot_name[i] + '.pdf'))
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
conc_plot(res_path, plot_name, phase, PyCHAM_path, csv_path, obs_plot, t_col_indx, m_col_indx)