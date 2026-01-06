# concentration_analysis

## Version 0.1.1

Analyse simulated concentrations.

Example outputs of AtCHEM2, INCHEM-Py and PyCHAM are stored in src/concentration_analysis/example_outputs, with eponymous file names respective of source model.

Plotting and statistical tools to work on simulated concentrations, all modules require user input through editing of the user-definition sections:

* atomicn_generator returns a file containing the number of atoms per component
* CdC_reporter returns the concentration and rate of change (with respect to time) of concentration of components at the end of the simulation
* comp_names_file_rename stores the component names in a file with a name of the user's choosing
* ind_component plots the temporal profile(s) of concentration of user-specified component(s)
* MLM_generator_for_random_sample uses AutoML to create a machine learnt model for predicting secondary organic aerosol mass yields based on provided training data
* NO_HO2_RO2pool_reporter returns the concentrations of NO, HO2 and the RO2 pool at the end of the simulation
* particle_phase_contributors plots the temporal profiles of mass fractions of individual components in the particle phase
* total_particle_mass_from_number_size_distribution plots particle mass based on the particle number distribution across particle size bins
* total_particle_vs_obs allows comparison of any number of simulated total particle concentrations against observations
* SOA_yield returns the secondary organic aerosol mass yield (mass of SOA/mass of consumed VOC), in addition to any specified parameters, for any number of simulations 

The src/package layout is explained [here](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-structure.html)

For MLM_generator_for_random_sample.py:

#### AutoML and Machine Learning Enhancements
- Improved AutoML usage model selection across ["xgboost", "rf", "lgbm", "catboost", "extra_tree", 
                               "kneighbor", "enet", "histgb", "sgd"]
- Added multiple optimisation metrics to iterate through (RMSE, MAE, MSE, MAPE, R²) with automated model selection
- Integrated XGBoost for quantile regression to estimate prediction uncertainties after finding that it performed best according to AutoML.

#### Data Processing and Transformation
- Added robust data transformation pipeline including:
  - Automatic handling of non-finite values
  - Log transformation for skewed data
  - Power transformation using Yeo-Johnson method
  - Min-max scaling for feature normalisation
- Implemented automatic train-test splitting with reproducible results

#### Time Series Analysis
- Added time series prediction capabilities for SOA yields 
- Implemented spatial dimension support for predictions
- Added picklebatchdivider.py script to handle day-wise and hour-wise data processing

#### Uncertainty Quantification
- Implemented quantile regression (5th, 50th, 95th percentiles) for uncertainty bounds
- Added mortality impact calculations using Havala Pye's coefficients - doi: 10.1038/s41467-021-27484-1
- Developed comprehensive uncertainty visualisation tools

#### Visualisation and Reporting
- Enhanced visualisation suite including:
  - Distribution plots for all predictions
  - Quantile regression prediction intervals
  - Model performance comparison charts
  - Uncertainty visualisation across time series
- Added detailed statistical reporting including NRMSE, Pearson correlation, and MAE

### Installation and Usage
See the original documentation above for basic usage. For new features:
1. Run pip install -r requirements.txt if on Windows to quickly get all needed packages. See all required packages in requirements.txt.

2. If testing different datasets on the model, make sure to put pickle files of the following parameters:
['SOA mass yield (fraction)', 'Temperature (K)', 'Pressure (Pa)', 'J(NO2) (/s)', 'Total absorbing mass of PM (ug/m^3)', 'APINENE (molecules/cm^3)', 'BENZENE (molecules/cm^3)', 'NO (molecules/cm^3)', 'HO2 (molecules/cm^3)', 'CH3O2 (molecules/cm^3)', 'O3 (molecules/cm^3)', 'OH (molecules/cm^3)', 'NO3 (molecules/cm^3)', 'O:C'] 
Modify path at the top of picklebatchdivider.py as needed.

3. AutoML model training: Use option 1 in the interactive menu.
4. Uncertainty analysis: Use option 2 for quantile regression.
5. Time series predictions: Use options 4 and 5 sequentially.
