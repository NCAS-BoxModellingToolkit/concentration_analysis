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
import platform

# user-defined variables start --------------------------------

# set base path, depending on operating system
if 'Darwin' in platform.system() or 'Linux' in platform.system():
	base_path = str('/Users/user/Library/CloudStorage/' +
			'OneDrive-TheUniversityofManchester/')
if 'Win' in platform.system() or 'Linux' in platform.system():
	base_path = 'C:/Users/Psymo/OneDrive - The University of Manchester/'

# set list of path to results
res_path = [str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-2w_mt2p40_NANNOOLAL_pcrh'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-2w_mt24p0_NANNOOLAL_pcrh'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-3w_mt2p40_NANNOOLAL_pcrh'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-3w_mt24p0_NANNOOLAL_pcrh'),  
	str(base_path + 'NCAS/' +
	'MCM_working_group/guaiacol/PyCHAM_output/' +
	'1e-4w_mt2p40_NANNOOLAL_pcrh')]

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

	# open observed concentration
	wb = np.loadtxt(csv_path, delimiter = ',', skiprows = 1, dtype='str')
	# get observed time through experiment and particle mass concentration
	obs_pm_thr = wb[:, t_col_indx].astype('float')
	obs_pm_mass = wb[:, m_col_indx].astype('float')

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

		# particle-phase concentrations of all components (# molecules/cm^3)
		ppc = yrec[:, self.ro_obj.nc:-self.ro_obj.nc*self.ro_obj.wf]

		# sum individual components over particle size bins (# molecules/cm^3)
		for psbi in range(1, self.ro_obj.nsb-self.ro_obj.wf):
			ppc[:, 0:nc] += ppc[:, 
				(psbi)*nc:(psbi+1)*nc]

		# convert # molecules/cm3 to moles/m^3
		ppc[:, 0:nc] = (ppc[:, 0:nc]/si.N_A)*1.e6

		# zero water
		ppc[:, self.ro_obj.H2O_ind] = 0.

		# convert moles/m3 to ug/m^3
		ppc[:, 0:nc] = ppc[:, 0:nc]*y_MM*1.e6

		# sum over components for total dry particle mass concentration
		# over time (ug/m^3)
		ppc = np.sum(ppc[:, 0:nc], axis=1)

		thr -= 1. # remove spin-up time

		ax0.plot(thr[thr>=0.], ppc[thr>=0.], label = labels[resi])

		# using just the simulation results within observed time,
		# interpolate simulated results to times of observations
		t_indx = (thr>=obs_pm_thr[0])*(thr<=obs_pm_thr[-1])

		ppc_sim = np.interp(obs_pm_thr, thr[t_indx], ppc[t_indx])

		# root mean square error
		rmse = (np.sum(((obs_pm_mass-ppc_sim)**2.))/len(obs_pm_mass))**0.5

		# state root mean square error beside plot
		ax0.text(max(thr), ppc[-1], str(round(rmse, 0)), fontsize = 14)

	ax0.plot(obs_pm_thr, obs_pm_mass, 'k', label = 'observed')
	print(min(obs_pm_thr))
	ax0.set_xlim(left=-0.05, right=1.6)

	ax0.set_ylabel(str('PM mass concentration (anhydrous) / '  + 
		'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3'), fontsize = 14)
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
conc_plot(res_path, labels, conc_to_plot, PyCHAM_path, plot_name, save_path, 
	csv_path, t_col_indx, m_col_indx)