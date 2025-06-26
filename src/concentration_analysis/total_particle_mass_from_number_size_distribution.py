'''code to plot simulated total particle mass concentration from
simulated number size distributions of particle'''
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
	'guaiacol_constrained_1e-3w_mt2p40_fullELVOC_120sb')]

# set corresponding (to path to results) list of plot labels
labels = [
str('$C_w\mathrm{=1x10^{-3}\; g \, m\u207B\u00B3}$, $k_e$=4x10$^{-1}\,\mathrm{s^{-1}}$')]

# concentration(s) to plot (m for mass concentration)
conc_to_plot = ['m']

# name of plot
plot_name = 'dry_pm_mass_fromN_vs_time'

# path to save plot to
save_path = str(base_path + 'NCAS/MCM_working_group/guaiacol/PyCHAM_output')

# state the assumed density of particle components (g/cm^3)
rho = 1.

# time (hour) through experiment to treat as zero and therefore start
# plotting results at
tstart = 1.

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, labels, conc_to_plot, plot_name, save_path, rho, tstart):

	# prepare plot(s)
	fig, (ax0) = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

	resi = -1 # count on simulation results

	# loop through simulations
	for res_pathi in res_path:

		resi += 1 # count on simulation results

		# create the self object so that results path is stored
		self = self_def(res_pathi)

		# import results, time (hour), size bin bound radius (um),
		# number concentration per size bin of dry particles (particles/cm^3)
		[thr, rbou_rec, Ndry] = var_get(self)

		# get radius of particles (cm) at the centre of bin bounds
		r = (rbou_rec[0, 0:-1]+(rbou_rec[0, 1::]-rbou_rec[0, 0:-1])/2.)*1.e-4

		# volume of single particles (cm^3) at bin centre
		V = (4./3.)*np.pi*r**3

		# mass of single particles at bin centre (ug), tiled over times
		m = np.tile(((V*rho)*1.e6).reshape(1, -1), (len(thr), 1))

		# mass of all particles per size bin (ug/m^3)
		m = (m*Ndry)*1.e6

		# mass of particles summed over size bins (ug/m^3)
		m = np.sum(m, axis=1)

		ax0.plot(thr[thr>=tstart]-tstart, m[thr>=tstart], 
			label = labels[resi])

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

# function to retrieve variables
def var_get(self):

	# withdraw times (s)
	fname = str(self.dir_path + '/time')
	t_array = np.loadtxt(fname, delimiter=',', skiprows=1)
	thr = t_array/3600.0 # convert from s to hr

	# particle size bin bounds (radii) (um3)
	fname = str(self.dir_path + '/size_bin_bounds')
	rbou_rec = np.loadtxt(fname, delimiter=',', skiprows=1)

	# withdraw number-size distributions (# particles/cm3 (air))
	fname = str(self.dir_path + '/particle_number_concentration_dry')
	Ndry = np.loadtxt(fname, delimiter=',', skiprows=1)
	
	return(thr, rbou_rec, Ndry)

# call function
conc_plot(res_path, labels, conc_to_plot, plot_name, save_path, rho, tstart)