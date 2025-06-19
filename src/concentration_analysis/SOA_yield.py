'''code to estimate secondary organic aerosol '''
'''mass yield (mass of SOA/mass of consumed VOC), '''
'''and return any specified parameters, for any '''
'''number of simulations '''

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import os
import openpyxl # for opening excel file
import netCDF4 as nc
import matplotlib.pyplot as plt
import pickle

# user-defined variables start --------------------------------

# set list of path to results
res_path = str('/Users/user/Library/CloudStorage/OneDrive-TheUnivers' +
	'ityofManchester/SOAPRA/EMEP/PyCHAM_outputs/' +
	'w_and_wo_RO2int/random_sampling')

# names of gas-phase VOCs to consider for estimating SOA mass yield
VOCs = ['APINENE', 'BENZENE']

# molar masses of VOCs (g/mol), must align with VOC names (VOCs)
VOC_MM = [136.24, 78.11]

# path to save results to
save_path = str('/Users/user/Library/CloudStorage/OneDrive-' +
	'TheUniversityofManchester/SOAPRA/EMEP/stats/random_sampling/SOA_mass_yields.pickle')

# user-defined variables end --------------------------------

# define function
def SOA_mass_yield(res_path, VOCs, VOC_MM, save_path):

	# get names of folders containing results
	paths_to_res_files = os.listdir(res_path)

	# remove DS_store file from list
	paths_to_res_files.remove('.DS_Store')

	# remove any simulations without a given process
	for pi in paths_to_res_files:
		if '_wo' in pi:
			paths_to_res_files.remove(pi)

	# sort in alphabetical/numerical order
	paths_to_res_files.sort()

	# prepare to hold the SOA mass yield (fraction)
	# (mass of SOA formed/mass of VOC consumed)
	SOAmy = {}

	# hold names of variables in first dictionary
	SOAmy['var_names'] = ['SOA mass yield (fraction)', 'Temperature (K)', 'Pressure (Pa)', 
		'J(NO2) (/s)', 'Total absorbing mass of PM (ug/m^3)', 
		'APINENE (molecules/cm^3)', 'BENZENE (molecules/cm^3)', 
		'NO (molecules/cm^3)', 'HO2 (molecules/cm^3)', 'CH3O2 (molecules/cm^3)',
		'O3 (molecules/cm^3)', 'OH (molecules/cm^3)', 'NO3 (molecules/cm^3)',
		'O:C']

	# loop through folders
	for resi in range(len(paths_to_res_files)):

		# get recorded simulation number
		recn = paths_to_res_files[resi]
		recn = recn[0:recn.index('_')]

		# suffix name of netCDF file
		res_path_w = str(res_path + '/' + paths_to_res_files[resi] + 
			'/PyCHAM_results.nc' )

		ds = nc.Dataset(res_path_w) # open file
		
		# get SOA mass concentration (ug/m^3) at different times
		SOA_w = np.array((ds.variables[str('mass_concentration_of_secondary_' + 
		'particulate_organic_matter_dry_aerosol_particles_in_air')]))

		# get incrememnt in SOA mass concentration (ug/m^3) at different times
		SOA_w[1::] = SOA_w[1::]-SOA_w[0:-1]
	
		# prepapre to hold number concentrations of VOCs
		VOC_c = np.zeros((len(SOA_w), len(VOCs)))
		
		# time though simulation (s)
		t = np.array((ds.variables['time']))

		for vi in range(len(VOCs)):

			# string to use for retrieving result
			strn = str('number_concentration_of_gas_phase_' + 
				VOCs[vi] + '_molecules_in_air')

			# get VOC concentrations (molecules/cm^3) at different times
			VOC_c[:, vi] = np.array((ds['/concentrations_g'].variables[
				strn]))

			# convert gas-phase concentrations from molecules/cm^3 to ug/m^3
			VOC_c[:, vi] = ((VOC_c[:, vi]/si.N_A)*VOC_MM[vi])*1.e12
			
			# mass of VOC consumed at different times (ug/m^3)
			VOC_c[1::, vi] = (VOC_c[0:-1, vi]-VOC_c[1::, vi])

		# sum of VOC consumed at different times (ug/m^3)
		VOC_c = np.sum(VOC_c, axis=1)

		# get the number of time points where consumption of VOC exceeds 1 %
		# of the maximum VOC consumption for this simulation
		ti = VOC_c>0.01*(max(VOC_c))
		tn = sum(ti)
		if tn == 0:
			import ipdb; ipdb.set_trace()
		# prepare to hold parameters and outcome (rows) against time (columns)
		SOAmy[recn] = np.zeros((tn, 14))

		# SOA mass yield (fraction), the ouctome
		SOAmy[recn][:, 0] = (SOA_w[ti])/(VOC_c[ti])
		# record parameters in later rows
		SOAmy[recn] = param_rec(ds, SOAmy[recn], ti)

		ds.close() # close file
	
	# save SOA mass yields
	with open(save_path, 'wb') as handle:
		pickle.dump(SOAmy, handle)

# function to record parameters
def param_rec(ds, SOAmy, ti):
	
	SOAmy[:, 1] = np.array((ds.variables['temperature']))[ti]

	SOAmy[:, 2] = np.array((ds.variables['air_pressure']))[ti]

	SOAmy[:, 3] = np.array((ds.variables['transmission factor of light']))[ti]*0.006779

	SOAmy[:, 4] = np.array((ds.variables[
			str('mass_concentration_of_' + 
		'particulate_matter_wet_aerosol_particles_in_air')]))[ti]

	SOAmy[:, 5] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_APINENE' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 6] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_BENZENE' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 7] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_NO' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 8] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_HO2' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 9] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_CH3O2' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 10] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_O3' + 
				'_molecules_in_air')]))[ti]
	
	SOAmy[:, 11] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_OH' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 12] = np.array((ds['/concentrations_g'].variables[
			str('number_concentration_of_gas_phase_NO3' + 
				'_molecules_in_air')]))[ti]

	SOAmy[:, 13] = np.array((ds.variables[
			str('O:C_for_C>4_weighted_by_number' + 
				'_concentration_in_air')]))[ti]

	return(SOAmy)

# call function
SOA_mass_yield(res_path, VOCs, VOC_MM, save_path)