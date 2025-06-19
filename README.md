# concentration_analysis
Analyse simulated concentrations.

Plotting and statistical tools to work on simulated concentrations, all modules require user input through editing of the user-definition sections:

* atomicn_generator returns a file containing the number of atoms per component
* CdC_reporter returns the concentration and rate of change (with respect to time) of concentration of components at the end of the simulation
* comp_names_file_rename stores the component names in a file with a name of the user's choosing
* ind_component plots the temporal profile(s) of concentration of user-specified component(s)
* MLM_generator_for_random_sample uses AutoML to create a machine learnt model for predicting secondary organic aerosol mass yields based on provided training data
* NO_HO2_RO2pool_reporter returns the concentrations of NO, HO2 and the RO2 pool at the end of the simulation
* total_particle_vs_obs allows comparison of any number of simulated total particle concentrations against observations
* SOA_yield returns the secondary organic aerosol mass yield (mass of SOA/mass of consumed VOC), in addition to any specified parameters, for any number of simulations 


The src/package layout is explained [here](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)

Version 1.0.0
