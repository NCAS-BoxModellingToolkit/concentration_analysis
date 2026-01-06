'''code to plot simulated contributions of individual 
components to the particle phase'''

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import matplotlib.pyplot as plt
import os
import platform

# user-defined variables start --------------------------------

# set base path, depending on operating system
if 'Darwin' in platform.system() or 'Linux' in platform.system():
	base_path = str('/Users/user/Library/CloudStorage/' +
			'OneDrive-TheUniversityofManchester/')
if 'Win' in platform.system() or 'Linux' in platform.system():
	base_path = 'C:/Users/Psymo/OneDrive - The University of Manchester/'

# set path to results
res_path = [str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-2w_mt24p0_NANNOOLAL_pcrh')]

# path to PyCHAM
PyCHAM_path = str(base_path + 'GitHub/PyCHAM/PyCHAM')

# name of plot
plot_name = 'dry_pm_mass_frac_contributors_vs_time'

# path to save plot to
save_path = str(base_path + 'NCAS/MCM_working_group/guaiacol/PyCHAM_output')

# whether (0) or not (1) to include water in particle-phase
# mass fraction calculation
anhydrous_flag = 1

# number of greatest contributing components to time-integrated
# particle-phase mass concentration to plot (e.g. 6 is the top 
# 6 contributors)
top_num = 6

# user-defined variables end --------------------------------

# define function
def cont_plot(res_path, PyCHAM_path, plot_name, save_path, 
			anhydrous_flag, top_num):

	# prepare plot(s)
	fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

	# ensure PyCHAM can be seen
	sys.path.append(PyCHAM_path)
	import retr_out

	resi = -1 # count on simulation results

	# loop through simulations
	for res_pathi in res_path:

		resi += 1 # count on simulation results

		# create the self object so that results path is stored
		self = self_def(res_pathi)

		# import results
		try:
			for prog in retr_out.retr_out(self):
				prog = prog
		except:
			import ipdb; ipdb.set_trace()
			continue

		# get concentrations
		yrec = np.zeros((self.ro_obj.yrec.shape[0], 
			self.ro_obj.yrec.shape[1]))
		yrec[:, :] = self.ro_obj.yrec[:, :]

		# get molar masses of component (g/mol)
		y_MM = np.array((self.ro_obj.comp_MW)).reshape(1, -1)

		# get time (hours) through simulation
		thr = self.ro_obj.thr

		# get number of components
		nc = self.ro_obj.nc

		# get names of components
		comp_names = self.ro_obj.names_of_comp

		# particle-phase concentrations of all components (# molecules/cm^3)
		ppc = yrec[:, self.ro_obj.nc:-self.ro_obj.nc*self.ro_obj.wf]

		# sum individual components over particle size bins (# molecules/cm^3)
		for psbi in range(1, self.ro_obj.nsb-self.ro_obj.wf):
			ppc[:, 0:nc] += ppc[:, 
				(psbi)*nc:(psbi+1)*nc]

		# convert # molecules/cm3 to moles/m^3
		ppc[:, 0:nc] = (ppc[:, 0:nc]/si.N_A)*1.e6
		
		if (anhydrous_flag == 1):
			# zero water
			ppc[:, self.ro_obj.H2O_ind] = 0.

		# convert moles/m^3 to ug/m^3
		ppc[:, 0:nc] = ppc[:, 0:nc]*y_MM*1.e6

		# sum over components for total dry particle mass concentration
		# over time (ug/m^3)
		ppc_tot = (np.sum(ppc[:, 0:nc], axis=1)).reshape(-1, 1)

		# prepare to hold particle-phase mass fractions of
		# each component over time (%)
		ppmf = np.zeros((ppc.shape[0], nc))

		# non-zero indices
		nzi = ppc_tot[:, 0] != 0.

		# get particle-phase mass fractions of each component over time
		# (%)
		ppmf[nzi, :] = (ppc[nzi, 0:nc]/ppc_tot[nzi, :])*100.

		# get order of components in ascending order of their
		# time-integrated particle-phase mass fraction
		sort_index = np.argsort(np.sum(ppmf, axis=0))

		# only include times when roof open
		tindx = (thr >= 1.)

		# loop through components in descending order of
		# their time-integrated particle phase mass fraction
		for ci in range(1, top_num+1):
			
			ax0.plot(thr[tindx]-1., ppmf[tindx, sort_index[-(ci)]], label = comp_names[sort_index[-(ci)]])

	if (anhydrous_flag == 1):
		ax0.set_ylabel(str('PM mass fraction (anhydrous) / %'), 
				 fontsize = 14)
	ax0.set_xlabel(str('Time'), fontsize = 14)
	ax0.yaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.xaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.legend()

	# replace time through simulation with clock time on abscissa
	xticks = [0., 0.5, 1., 1.5]
	xlabels = ['08:30', '09:00', '09:30', '10:00']
	ax0.set_xticks(xticks, labels=xlabels)
	
	# show grid lines
	ax0.grid(visible=True, which='major', axis='both')

	plt.tight_layout()
	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))
	plt.savefig(str(save_path + '/' + plot_name + '.pdf'))
	
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
cont_plot(res_path, PyCHAM_path, plot_name, save_path, 
		  anhydrous_flag, top_num)