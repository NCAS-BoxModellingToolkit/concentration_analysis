'''code to plot simulated concentrations of total particulate
matter and compare against observations'''
# concentrations may be from any number of simulations

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import matplotlib.pyplot as plt
import os
import openpyxl # for opening excel file

# user-defined variables start --------------------------------

# set list of path to results
res_path = [str('/Users/user/Library/CloudStorage/' +
	'OneDrive-TheUniversityofManchester/NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-3w'), str('/Users/user/Library/CloudStorage/' +
	'OneDrive-TheUniversityofManchester/NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-1w')]

# set corresponding (to path to results) list of plot labels
labels = [str('$\mathrm{C_w=1x10^3\; \u00B5 g \, m\u207B\u00B3}$'), str('$\mathrm{C_w=1x10^5\; \u00B5 g \, m\u207B\u00B3}$')]

# concentration(s) to plot (m for mass concentration)
conc_to_plot = ['m']

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# name of plot
plot_name = 'dry_pm_mass_vs_time'

# path to save plot to
save_path = str('/Users/user/Library/CloudStorage/OneDrive-' +
	'TheUniversityofManchester/NCAS/MCM_working_group/guaiacol/PyCHAM_output')

# path to observations in csv file
csv_path = str('/Users/user/Library/CloudStorage/OneDrive-TheUniversity' +
	'ofManchester/NCAS/MCM_working_group/guaiacol/' +
	'SMPS_total_N_SA_V_Mass_for_MCM_corrected.csv')

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, labels, conc_to_plot, PyCHAM_path, plot_name, save_path, csv_path):

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
		for prog in retr_out.retr_out(self):
			prog = prog

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

		# particle-phase concentrations of all components (# molecules/cm3)
		ppc = yrec[:, self.ro_obj.nc:-self.ro_obj.nc*self.ro_obj.wf]

		# sum individual components over particle size bins (# molecules/cm3)
		for psbi in range(1, self.ro_obj.nsb-self.ro_obj.wf):
			ppc[:, 0:nc] += ppc[:, 
				(psbi)*nc:(psbi+1)*nc]

		# convert # molecules/cm3 to moles/m3
		ppc[:, 0:nc] = (ppc[:, 0:nc]/si.N_A)*1.e6

		# zero water
		ppc[:, self.ro_obj.H2O_ind] = 0.

		# convert moles/m3 to ug/m3
		ppc[:, 0:nc] = ppc[:, 0:nc]*y_MM*1.e6

		# sum over components for total dry particle mass concentration
		# over time (ug/m3)
		ppc = np.sum(ppc[:, 0:nc], axis=1)

		ax0.plot(thr, ppc, label = labels[resi])

	# open observed concentration
	wb = np.loadtxt(csv_path, delimiter = ',', skiprows = 1, dtype='str')
	# get observed time through experiment and particle mass concentration
	obs_pm_thr = wb[:, 1].astype('float')
	obs_pm_mass = wb[:, 9].astype('float')

	ax0.plot(obs_pm_thr, obs_pm_mass, label = 'observed')

	ax0.set_ylabel(str('PM mass concentration ('  + 
		'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)\n(no water)'), fontsize = 14)
	ax0.set_xlabel(str('time through simulation (hours)'), fontsize = 14)
	ax0.yaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.xaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.legend()
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
conc_plot(res_path, labels, conc_to_plot, PyCHAM_path, plot_name, save_path, csv_path)