# concentration_analysis
Analyse simulated concentrations.

Plotting and statistical tools to work on simulated concentrations, all modules require user input through editing of the user-definition sections:

* atomicn_generator returns a file containing the number of atoms per component
* CdC_reporter returns the concentration and rate of change (with respect to time) of concentration of components at the end of the simulation
* comp_names_file_rename stores the component names in a file with a name of the user's choosing
* ind_component plots the temporal profile(s) of concentration of user-specified component(s)
* NO_HO2_RO2pool_reporter returns the concentrations of NO, HO2 and the RO2 pool at the end of the simulation
* total_particle_vs_obs allows comparison of total particle concentrations against observations 


The src/package layout is explained [here](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)
