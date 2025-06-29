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

# start of paths
base_path = 'C:/Users/Psymo/OneDrive - The University of Manchester/'
# set list of path to results
res_path = [str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-2w_mt2p40_fullELVOC'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-2w_mt24p0_fullELVOC'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-3w_mt2p40_fullELVOC'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-3w_mt24p0_fullELVOC'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'guaiacol_constrained_1e-4w_mt2p40_fullELVOC')]

# set corresponding (to path to results) list of plot labels
labels = [
str('$C_w\mathrm{=1x10^{-2}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-1}\,\mathrm{s^{-1}}$'),
str('$C_w\mathrm{=1x10^{-2}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-2}\,\mathrm{s^{-1}}$'),
str('$C_w\mathrm{=1x10^{-3}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-1}\,\mathrm{s^{-1}}$'),
str('$C_w\mathrm{=1x10^{-3}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-2}\,\mathrm{s^{-1}}$'), 
str('$C_w\mathrm{=1x10^{-4}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-1}\,\mathrm{s^{-1}}$')]

# concentration(s) to plot (m for mass concentration)
conc_to_plot = ['m']


# path to PyCHAM
PyCHAM_path = str(base_path + 'GitHub/PyCHAM/PyCHAM')

# name of plot
plot_name = 'dry_pm_mass_vs_time'

# path to save plot to
save_path = str(base_path + 'NCAS/MCM_working_group/guaiacol/PyCHAM_output')

# path to observations in csv file
csv_path = str(base_path + 'NCAS/MCM_working_group/guaiacol/' +
	'SMPS_total_N_SA_V_Mass_for_MCM_corrected.csv')

# column of observations file containing times
t_col_indx = 1
# column of observations file containing particle mass concentrations
m_col_indx = 9

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, labels, conc_to_plot, PyCHAM_path, plot_name, save_path, 
	csv_path, t_col_indx, m_col_indx):

	# prepare plot(s)
	fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

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

		if 'fullELVOC' in res_pathi:
			thr -= 1.

		ax0.plot(thr[thr>=0.], ppc[thr>=0.], label = labels[resi])

	# open observed concentration
	wb = np.loadtxt(csv_path, delimiter = ',', skiprows = 1, dtype='str')
	# get observed time through experiment and particle mass concentration
	obs_pm_thr = wb[:, t_col_indx].astype('float')
	obs_pm_mass = wb[:, m_col_indx].astype('float')

	ax0.plot(obs_pm_thr, obs_pm_mass, 'k', label = 'observed')

	ax0.set_ylabel(str('PM mass concentration ('  + 
		'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)\n(no water)'), fontsize = 14)
	ax0.set_xlabel(str('Time since lights on (hours)'), fontsize = 14)
	ax0.yaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.xaxis.set_tick_params(labelsize = 14, 
		direction = 'in', which='both')
	ax0.legend()
	plt.tight_layout()
	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))
	plt.savefig(str(save_path + '/' + plot_name + '.png'))
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
conc_plot(res_path, labels, conc_to_plot, PyCHAM_path, plot_name, save_path, 
	csv_path, t_col_indx, m_col_indx)