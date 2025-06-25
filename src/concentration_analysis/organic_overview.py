'''code to plot simulated concentrations of total NMVOC,''' 
'''total oxidised non-methane organic in gas-phase and'''
'''total non-methane organic aerosol'''
# concentrations are plotted against time

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import matplotlib.pyplot as plt
import os

# user-defined variables start --------------------------------

# set path to results
res_path = [str('/Users/user/Library/CloudStorage/OneDrive-The' +
	'UniversityofManchester/SOAPRA/EMEP/PyCHAM_outputs/w_and' +
	'_wo_RO2int/random_sampling/plotting_sample/553_w'),
	str('/Users/user/Library/CloudStorage/OneDrive-The' +
	'UniversityofManchester/SOAPRA/EMEP/PyCHAM_outputs/w_and' +
	'_wo_RO2int/random_sampling/plotting_sample/2600_w'),
	str('/Users/user/Library/CloudStorage/OneDrive-The' +
	'UniversityofManchester/SOAPRA/EMEP/PyCHAM_outputs/w_and' +
	'_wo_RO2int/random_sampling/plotting_sample/5053_w')]

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# save_path
save_path = str('/Users/user/Library/CloudStorage/OneDrive-The' +
	'UniversityofManchester/SOAPRA/papers/AutoML/figs')

# names of VOCs to consider
NMVOC_names = ['APINENE', 'BENZENE']

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, PyCHAM_path, NMVOC_names, save_path):

	# ensure PyCHAM can be seeb
	sys.path.append(PyCHAM_path)
	import retr_out

	# prepare plots
	fig, axs = plt.subplots(nrows=1, ncols=len(res_path), figsize=(6, 3))

	ri = -1 # counter on results
	for resi in res_path: # loop through results

		ri += 1 # counter on results

		# title of sub-plot now
		titlen = resi[-resi[::-1].index('/'):-(resi[::-1].index('_')+1)]

		# create the self object so that results path is stored
		self = self_def(resi)
		
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

		# get O:C ratios of components
		O_to_C = np.array(self.ro_obj.O_to_C).reshape(-1)

		# number of particle size bins
		nsb = int(self.ro_obj.nsb)

		# prepare to hold gas-phase mass concentrations of NMVOCs (ug/m^3)
		gpNMVOC = np.zeros((len(thr)))

		# prepare to hold gas-phase mass concentrations of oxidised NMVOCs (ug/m^3)
		gpNMOOC = np.zeros((len(thr)))

		# prepare to hold particle-phase mass concentrations of NMVOCs (ug/m^3)
		ppNMOOC = np.zeros((len(thr)))

		# prepare to hold SOA mass yield (fraction)
		SOA_my = np.zeros((len(thr)))

		# factor to convert ppb to molecules/cm^3
		cfac = np.array(self.ro_obj.cfac).reshape(-1)

		# loop through components
		for i in range(nc):

			# check whether component is NMVOC
			if comp_names[i] in NMVOC_names:
				# convert ppb to ug/m^3 and sum
				gpNMVOC[:] += ((yrec[:, i]*cfac)/si.N_A)*y_MM[i]*1.e12
		
			if (O_to_C[i] > 0 and y_MM[i]>60.):
				# for gas phase convert ppb to ug/m^3 and sum
				gpNMOOC[:] += ((yrec[:, i]*cfac)/si.N_A)*y_MM[i]*1.e12
				# for particle phase loop through particle size bins and
				# convert molecules/cm^3 to ug/m^3
				for pi in range(1, nsb+1):
					ppNMOOC[:] += ((yrec[:, nc*pi+i])/si.N_A)*y_MM[i]*1.e12
		
		# mass of VOC consumed (ug/m^3)
		VOCcons = gpNMVOC[0:-1]-gpNMVOC[1::]
		yi = (VOCcons != 0.) # index of non-zero VOC consumed

		# mass of SOA generated (ug/m^3)
		SOAm = 	ppNMOOC[1::]-ppNMOOC[0:-1]

		# for each time step get the organic aerosol mass yield
		SOA_my[1::][yi] = (SOAm[yi])/(VOCcons[yi])

		# get time indices wanted to plot
		ti = (thr >= 3.)
		p0, = axs[ri].plot(thr[ti]-3., gpNMVOC[ti], 'k', 
			label = str('non-methane VOC precursor mass (' + 
			'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)'))

		p1, = axs[ri].plot(thr[ti]-3., gpNMOOC[ti], 'tab:orange', 
			label = str('non-methane oxidised ' +
			'organic gas mass (' + 
			'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)'))

		p2, = axs[ri].plot(thr[ti]-3., ppNMOOC[ti], 'aquamarine', 
			label = str('non-methane oxidised ' +
			'organic particle mass (' + 
			'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)'))

		
		par1 = axs[ri].twinx() #  parasite axis
		p3, = par1.plot(thr[ti]-3., SOA_my[ti], 'g', 
			label = str('aerosol mass yield ($\gamma$SOA)'))
		par1.set_ylim(bottom = -1.2, top = 1.2)

		axs[ri].set_title(str(titlen), fontsize = 8)
		if (ri == 0): # only need left vertical axis label on left-most plot
			axs[ri].set_ylabel(str('mass concentration ('  + 
				'$\mathrm{\u00B5}$g$\,$m\u207B\u00B3)'), fontsize = 8)
		axs[ri].set_xlabel(str('time (hours)'), fontsize = 8)
		axs[ri].yaxis.set_tick_params(labelsize = 8, 
			direction = 'in', which='both')
		axs[ri].xaxis.set_tick_params(labelsize = 8, 
			direction = 'in', which='both')

		par1.yaxis.set_tick_params(labelsize = 8, 
			direction = 'in', which='both')

		if (ri == len(res_path)-1):
			par1.set_ylabel('$\gamma$SOA', size=8, 
				rotation=270, labelpad=15) # vertical axis label
			
	# only need legend stated for one subplot
	plt.legend(handles = [p0, p1, p2, p3], fontsize = 6, 
		bbox_to_anchor=(1, 1), bbox_transform=fig.transFigure)

	plt.subplots_adjust(bottom = 0.15, wspace = 0.7, top=0.7, right = 0.88)
	
	# make directory if not already existing
	plt.savefig(str(save_path + '/' + 'organic_overview.pdf'))
		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# call function
conc_plot(res_path, PyCHAM_path, NMVOC_names, save_path)