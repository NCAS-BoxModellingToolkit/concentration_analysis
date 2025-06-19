'''code for applying XGBoost to PyCHAM data, 
helped by: https://www.datacamp.com/tutorial/xgboost-in-python and
https://github.com/m-edal/Time-series-analytics-course/tree/69c083fc5ebefc645fac1479b976ad6af886ba2a (practical 3 solutions), 
must be run from 'conda activate PyCHAM' environment to ensure 
that shap_tree.py module has had np.int changed to int'''

# dependencies
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb	
from sklearn.metrics import mean_squared_error
from sklearn.inspection import partial_dependence
import platform
import shap
import plotly.express as px
import netCDF4 as nc # for netcdf results
import cartopy.crs as ccrs # for plotting maps
import pickle
from flaml import AutoML
import pandas as pd
from sklearn.datasets import fetch_california_housing

# start of user inputs ---------------------------
# path to PyCHAM results (generated via /Users/user/Documents/GitHub/
# NCAS-BoxModellingToolkit/concentration_analysis/src/concentration_
# analysis/SOA_yield.py)
path_to_res = str('/Users/user/Library/CloudStorage/' +
		'OneDrive-TheUniversityofManchester/SOAPRA' +
		'/EMEP/stats/random_sampling/SOA_mass_yields.pickle')

# path to which to save the statistics of
# ML model evaluation
stats_save_path = str('/Users/user/Library/CloudStorage/' +
	'OneDrive-TheUniversityofManchester/SOAPRA/' +
	'EMEP/stats/random_sampling/ML_evaluation_stats.csv')

# path to file to save machine learnt model to
model_save_path = str('/Users/user/Library/CloudStorage/' +
	'OneDrive-TheUniversityofManchester/SOAPRA/' +
	'EMEP/stats/random_sampling/' +
	'ML_model.pkl')

# flag for whether to conduct and plot SHAP analysis
SHAP_flag = 0
# end of user inputs -----------------------------
	

# define function for creating decision tree model and 
# generating predictions for effect on SOPM from the 
# decision tree model
def MLM_generator(path_to_res, stats_save_path, model_save_path, SHAP_flag):

	# inputs: ----------------------------------
	# path_to_res - path to 0D results
	# stats_save_path - path to which to save the statistics of
	#	ML model evaluation
	# model_save_path - path to save XGBoost model to
	# ------------------------------------------		

	# names of variables (first dictionary)
	# and values of variables per simulation (subsequent 
	# dictionaries)
	with open(path_to_res, 'rb') as handle:
    		var_vals = pickle.load(handle)

	# loop through subsequent dictionaries to get predictors and outcome
	for ki in var_vals:
		if ki == 'var_names': # names of 3D-constrained variables
			head_list = var_vals[ki]
			# prepare to hold predictors
			x = np.zeros((0, len(head_list)-1))
			# prepare to hold outcome
			y = np.zeros((0))
		else: # simulation results
			# hold predictors
			x = np.append(x, var_vals[ki][:, 1::], axis=0)
			# hold outcome
			y = np.append(y, 
			np.squeeze(var_vals[ki][:, 0]).reshape(-1), axis=0)

	print('Number of data points: ', y.shape[0])
	
	
	X_train, x_test, y_train, y_test = train_test_split(x, y,
		random_state=1)
	print(str('Number of NaN values in X_train and y_train: '), 
		sum(sum(np.isnan(X_train))), sum(np.isnan(y_train)))

	# initiate AutoML
	automl = AutoML()
	# Specify automl goal and constraint
	automl_settings = {
   	 "time_budget": 30,  # in seconds
   	 "metric": "r2",
   	 "task": "regression",
   	 "log_file_name": "SOA_mass_yield.log",
	}
	
	# pass the training data to AutoML to create a machine learnt model through
	# automatic optimisation of several machine learning techniques
	automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
	# predict
	print(automl.predict(X_train))
	# Print the best model
	print(automl.model.estimator)
 
	# save the model
	with open(model_save_path, 'wb') as f:
    		pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

	# open model
	with open(model_save_path, 'rb') as f:
    		automl = pickle.load(f)

	# get predictions for test set
	best_preds_test = automl.predict(x_test)

	return()

def XGBoost_call(): # function for using XGBoost
	
	# create regression matrices (XGBoost class for 
	# storing datasets is called DMatrix)
	dtrain_reg = xgb.DMatrix(x_train, y_train)
	dtest_reg = xgb.DMatrix(x_test, y_test)

	# define hyperparameters
	# for using absolute error as the objective function
	# which is defined in the function above
	# for using root mean square error as the objective function
	params = {"objective": "reg:squarederror", "tree_method": "hist"}

	# number of boosting rounds
	n = 1000

	# prepare to hold cross-validation model results
	cvboosters = []

	# cross-validation for thorough testing of model
	results = xgb.cv(
		params, 
		dtrain_reg,
		num_boost_round=n,
		nfold=5,
		early_stopping_rounds=20,
		callbacks = [savebestmodel(cvboosters),])

	# if using root mean square error as the objective function
	best_rmse = results['test-rmse-mean'].min()

	# prepare for evaluation
	evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

	# prepare to hold root mean square errors of 
	# starting models from cv retrained in train
	rmse = np.zeros((len(cvboosters)))

	# loop through models output by xgb cv to train against 
	# full test set and test against test set
	for modeli in range(len(cvboosters)): 
		model = xgb.train(
			params=params,
			dtrain=dtrain_reg,
			num_boost_round=n,
			evals=evals,
			verbose_eval=0,
			early_stopping_rounds=50,
			xgb_model=cvboosters[modeli])

		# test output model on test set
		preds = model.predict(dtest_reg)
		rmse[modeli] = (mean_squared_error(y_test, preds, squared=False))

		# if new model has lowest rmse then save as best model
		if (rmse[modeli] == min(rmse[0:modeli+1])):
			best_preds_test = preds
			best_model = model

	# save the generated XGBoost decision tree model
	best_model.save_model(model_save_path)

	# error statistics for the test set
	# mean absolute error
	mae_all = ((sum(np.abs(best_preds_test-y_test)))/len(y_test))
	# mean relative error
	# non-zero index
	nz = y_test != 0.
	mre_all = (sum(np.abs((best_preds_test[nz]-y_test[nz])/y_test[nz]))/len(y_test[nz]))

	# error stats for the test set beyond absolute lowest 10 %
	# of determined variable
	thresh = np.max(np.abs(y_test))*0.1
	#print(thresh)
	
	# indices of y test beyond threshold
	indx = np.abs(y_test)>thresh

	mae_large = ((sum(np.abs(best_preds_test[indx]-y_test[indx])))/
		len(y_test[indx]))
	mre_large = (sum(np.abs((best_preds_test[indx]-y_test[indx])/
		y_test[indx]))/len(y_test[indx]))

	# contain statistics in one array
	stats_arr = (np.array((mae_all, mre_all, thresh, mae_large, 
		mre_large))).reshape(1, -1)
	# header for statistics
	stats_hea = str('mae all test, mre all test, >threshold for sample, '+ 
		'mae>threshold sample, mre>threshold sample')

	np.savetxt(stats_save_path, stats_arr, delimiter=',', header = stats_hea)
	
	# least-squares solution to predicted vs. simulated SOA yield
	A = np.vstack([best_preds_test, np.ones(len(y_test))]).T
	m, c = np.linalg.lstsq(A, y_test, rcond=None)[0]
	
	# plot of predicted vs. simulated (ug/m^3)
	fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9., 4.))

	ax0.plot(y_test, best_preds_test, '+', 
		label = 'individual data')
	ax0.plot(y_test, y_test, '--', 
		label = '1:1')
	ax0.plot((y_test), m*(y_test)+c, '--', 
		label = str('least-squares regression\n(y=' + str(np.around(m, decimals=2)) + 
		'x+' + str(np.around(c, decimals=4)) + ')'))
	ax0.set_xlabel(str('$\gamma$SOA$_{actu}$'), fontsize = 14)
	ax0.set_ylabel(str('$\gamma$SOA$_{pred}$'),fontsize = 14)
	ax0.yaxis.set_tick_params(labelsize = 14, direction = 'in', which='both')
	ax0.xaxis.set_tick_params(labelsize = 14, direction = 'in', which='both')
	ax0.legend()

	# least-squares solution to error vs. simulated SOA yield
	A = np.vstack([np.abs(best_preds_test), np.ones(len(y_test))]).T
	m, c = np.linalg.lstsq(A, np.abs(best_preds_test-y_test), rcond=None)[0]
	
	ax1.plot(np.abs(best_preds_test), np.abs(best_preds_test-y_test), '+', 
		label = 'individual data')
	ax1.plot(np.abs(best_preds_test), m*np.abs(best_preds_test)+c, '--', 
		label = str('least-squares regression\n(y=' + str(np.around(m, decimals=2)) + 
		'x+' + str(np.around(c, decimals=4)) + ')'))
	ax1.set_xlabel(str('|$\gamma$SOA$_{pred}$|'), fontsize = 14)
	ax1.set_ylabel(str('|$\gamma$SOA$_{pred}$-$\gamma$SOA$_{actu}$|'), fontsize = 14)
	ax1.yaxis.set_tick_params(labelsize = 14, direction = 'in', which='both')
	ax1.xaxis.set_tick_params(labelsize = 14, direction = 'in', which='both')
	ax1.legend()

	ax0.text(-0.1, 1., 'a)', fontsize = 14, transform=ax0.transAxes)
	ax1.text(-0.1, 1., 'b)', fontsize = 14, transform=ax1.transAxes)
	
	plt.tight_layout()
	plt.show()

	# plot of SHapley Additive exPlanations (SHAP) analysis plots: Lundberg 2017
	# (https://proceedings.neurips.cc/paper_files/paper/2017/file/
	# 8a20a8621978632d76c43dfd28b67767-Paper.pdf) and Lundberg 2020 
	# (https://doi.org/10.1038/s42256-019-0138-9)
	if (SHAP_flag == 1):
		explainer = shap.TreeExplainer(best_model)
		shap_values = explainer.shap_values(x_train)

		shap.summary_plot(shap_values, x_train, 
			feature_names = head_list[1::], 
			color_bar_label='Parameter value', show=False)
		ax = plt.gca() # take control of the plot
		ax.set_xlabel(str('SHAP value (impact on $\gamma$[SOA])'))
		ax.xaxis.set_tick_params(labelsize = 14, direction = 'in', 
			which = 'both')
		plt.show() # show plot

	return() # end of MLM_generator function

# --------------------------------------------------
class savebestmodel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters
    
    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return(model)

# call function to generate machine learnt model
MLM_generator(path_to_res, stats_save_path, model_save_path, SHAP_flag)