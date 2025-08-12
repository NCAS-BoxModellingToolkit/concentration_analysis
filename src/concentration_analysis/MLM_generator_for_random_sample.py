'''code for applying AutoML to PyCHAM data to find best regressor, then applying quantile regession with best regressor to find model's uncertainty. 

We use Havala Pye's values to calculate the attributable mortality and show uncertainty using her values along with the model uncertainty
from doi: 10.1038/s41467-021-27484-1 where she states "SOA was associated with 8.9 (95% CI: 6.0-12) additional deaths per 100,000 in population 
for a 1 µg m^3 increase in concentration"
'''

# dependencies
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from flaml import AutoML
import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
import xgboost as xgboost
import os
from pathlib import Path

# start of user inputs------------------------
# path to PyCHAM results (generated via /Users/user/Documents/GitHub/
# NCAS-BoxModellingToolkit/concentration_analysis/src/concentration_
# analysis/SOA_yield.py)
path_to_res = r'C:\Users\16siv\Documents\GDMRR\XGBoost Solution\concentration_analysis\src\concentration_analysis\SOA_mass_yields.pickle'

# path to which to save the statistics of ML model evaluation
stats_save_path = r'C:\Users\16siv\Documents\GDMRR\XGBoost Solution\concentration_analysis\auto_ml_evaluation\ML_evaluation_stats.csv'

# path to file to save machine learnt model to
model_save_path = r'C:\Users\16siv\Documents\GDMRR\XGBoost Solution\concentration_analysis\auto_ml_evaluation\ML_model.pkl'

# path for the summary table of different metric optimisations
summary_save_path = r'C:\Users\16siv\Documents\GDMRR\XGBoost Solution\concentration_analysis\auto_ml_evaluation\ML_optimisation_summary.csv'

# flag for whether to conduct and plot shap analysis
SHAP_flag = 0
# end of user inputs -----------------------------


def apply_transformations(X, y):
    '''applies a series of data transformations to prepare features (X) and target (y) for machine learning
    
    the transformation pipeline includes:
    1. handling non-finite values
    2. making data positive for logarithmic transformation
    3. applying log transformation
    4. applying power transformation
    5. normalising the data
    
    args:
        X: input features array
        y: target values array
    
    returns:
        tuple containing:
        - normalised X values
        - normalised y values
        - power transformer for y (needed for inverse transformation)
        - normaliser for y (needed for inverse transformation)
        - y offset value (needed for inverse transformation)
    '''
    print("\nApplying Data Transformations...")

    # store the minimum y value to calculate offset later
    y_original_min = y.min()

    # replace nan, positive infinity and negative infinity with zeros
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)

    # ensure all y values are positive by adding an offset if needed
    y_offset = 0
    if y_original_min < 0:
        y_offset = abs(y_original_min) + 1
        y = y + y_offset

    # make all X values positive for logarithmic transformation
    if X.min() <= 0:
        X = X - X.min() + 1
    
    # apply logarithmic transformation using log1p (log(1 + x))
    # this helps handle skewed data and preserves zero values
    X_log = np.log1p(X)
    y_log = np.log1p(y.reshape(-1, 1))

    # apply yeo-johnson power transformation to make data more gaussian-like
    power_transformer_X = PowerTransformer(method='yeo-johnson')
    power_transformer_y = PowerTransformer(method='yeo-johnson')
    X_pow = power_transformer_X.fit_transform(X_log)
    y_pow = power_transformer_y.fit_transform(y_log)

    # scale all features to the range [0, 1] using min-max scaling
    normaliser_X = MinMaxScaler()
    normaliser_y = MinMaxScaler()
    X_normalised = normaliser_X.fit_transform(X_pow)
    y_normalised = normaliser_y.fit_transform(y_pow).ravel()
    
    print("Transformations complete.")
    
    return X_normalised, y_normalised, power_transformer_y, normaliser_y, y_offset


def load_and_prepare_data(path_to_res):
    '''loads simulation data from a pickle file and prepares it for machine learning
    
    this function:
    1. loads the 0D results
    2. separates features (x) and target (y)
    3. applies necessary transformations
    4. splits data into training and test sets
    
    args:
        path_to_res: path to the pickle file containing simulation results
    
    returns:
        dictionary containing processed datasets and transformation objects
    '''
    print("\nLoading and Preparing Data...")

    # names of variables (first dictionary)
    # and values of variables per simulation (subsequent 
    # dictionaries)
    with open(path_to_res, 'rb') as handle:
        var_vals = pickle.load(handle)

    # initialise arrays for storing parameter names and values
    head_list = []
    x = None
    y = None
    if 'var_names' in var_vals: # names of 3D-constrained variables
        head_list = var_vals['var_names']
        # initalise empty arrays with correct dimensions
        x = np.zeros((0, len(head_list)-1))  # -1 because first column is target
        y = np.zeros((0))
    
    # iterate through the dictionary to extract features and target
    for ki, val in var_vals.items():
        if ki != 'var_names':
            # append all columns except first to x (features)
            x = np.append(x, val[:, 1::], axis=0)
            # append first column to y (target)
            y = np.append(y, np.squeeze(val[:, 0]).reshape(-1), axis=0)
            
    print('Number of data points: ', y.shape[0])
    
    print(head_list)

    # keep original data for later comparison
    x_orig, y_orig = x.copy(), y.copy()
    
    # apply data transformations (normalision, scaling, etc.)
    x_normalised, y_normalised, power_transformer_y, normaliser_y, y_offset = apply_transformations(x, y)
    
    # split data into training and test sets with fixed random seed for reproducibility
    X_train_normalised, x_test_normalised, y_train_normalised, y_test_normalised = train_test_split(
        x_normalised, y_normalised, random_state=1
    )
    # split original data to keep corresponding test set in original scale
    _, _, _, y_test_orig = train_test_split(x_orig, y_orig, random_state=1)
    
    # package all processed data and transformation objects into a dictionary
    data_bundle = {
        "X_train": X_train_normalised,      # training features
        "x_test": x_test_normalised,        # test features
        "y_train": y_train_normalised,      # training targets
        "y_test": y_test_normalised,        # test targets
        "y_test_orig": y_test_orig,         # original scale test targets
        "power_transformer_y": power_transformer_y,  # for inverse transforming y
        "normaliser_y": normaliser_y,       # for inverse transforming y
        "y_offset": y_offset,               # for inverse transforming y
        "head_list": head_list              # names of variables
    }
    return data_bundle


def plot_automl_summary_chart(df_summary):
    '''creates a bar chart comparing different optimisation metrics and their performance scores
    
    the chart shows two metrics side by side:
    1. NRMSE (normalised Root Mean Square Error)
    2. r (correlation coefficient)
    
    args:
        df_summary: pandas DataFrame containing optimisation results and performance metrics
    '''
    # create a new figure with specified size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # extract metric names and create x-axis positions
    metrics = df_summary['optimised metric']
    x = np.arange(len(metrics))
    width = 0.35  # width of each bar
    
    # create two sets of bars for NRMSE and r scores
    rects1 = ax.bar(x - width/2, df_summary['NRMSE'], width, 
                     label='NRMSE', color='skyblue')  # left bars
    rects2 = ax.bar(x + width/2, df_summary['r'], width, 
                     label='r', color='salmon')       # right bars

    # customize the plot appearance
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance by Optimisation Metric')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # add value labels on top of each bar
    ax.bar_label(rects1, padding=3, fmt='%.4f')  # show 4 decimal places
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    
    # set y-axis limits to ensure all labels are visible
    ax.set_ylim(0, 1.1)
    
    # adjust layout to prevent overlapping
    fig.tight_layout()

    # show plot
    plt.show()


def run_automl_training(data_bundle):
    '''performs automated machine learning using FLAML to find the best model
    
    this function:
    1. tries multiple optimisation metrics
    2. trains models using different algorithms
    3. evaluates and compares their performance
    4. saves the best model and results
    
    args:
        data_bundle: dictionary containing training and test data
    
    returns:
        the best performing model estimator
    '''
    print("\nStarting AutoML Model Training")
    
    # extract training and test data from the data bundle
    X_train, y_train = data_bundle["X_train"], data_bundle["y_train"]
    x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]

    # list of metrics to try during optimisation
    metrics_to_optimise = ["rmse", "mae", "mse", "mape", "r2"]
    all_results = []

    print("Optimising using the following metrics:", metrics_to_optimise)
    
    # try each metric one by one
    for metric in metrics_to_optimise:
        print(f"\nOptimising for: {metric.upper()}")
        
        # create new AutoML instance
        automl = AutoML()
        
        # configure AutoML settings
        automl_settings = {
            "time_budget": 700,      # maximum time in seconds for training
            "max_iter": 200,          # maximum number of iterations
            "metric": metric,         # current optimisation metric
            "task": "regression",     # specify this is a regression task
            "log_file_name": f"SOA_mass_yield_{metric}_optim.log",   # log file name
            # list of ML algorithms to try
            "estimator_list": ["xgboost", "rf", "lgbm", "catboost", "extra_tree", 
                               "kneighbor", "enet", "histgb", "sgd"],
        }
        
        # train models using current metric
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        print(f"Best model found when optimising for {metric.upper()}: {automl.best_estimator}")

        # evaluate the best model found
        y_pred = automl.model.estimator.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        y_range = np.max(y_test) - np.min(y_test)
        nrmse = rmse / y_range if y_range != 0 else 0
        r_value = np.corrcoef(y_test, y_pred)[0, 1]

        # store results for this metric
        all_results.append({
            "optimised metric": metric.upper(),
            "Best Model Name": automl.best_estimator,
            "NRMSE": nrmse,
            "r": r_value,
            "automl_instance": automl,
        })
    
    # create summary dataframe of all results
    df_summary = pd.DataFrame(all_results)
    
    # display and save results
    print("\nSummary of All Optimisation Runs:")
    print(df_summary[['optimised metric', 'Best Model Name', 'NRMSE', 'r']])
    
    # save summary to csv
    df_summary[['optimised metric', 'Best Model Name', 'NRMSE', 'r']].to_csv(summary_save_path, index=False)
    print(f"\nOptimisation summary table saved to {summary_save_path}")

    # create visual summary
    plot_automl_summary_chart(df_summary)
    
    # select the model with lowest NRMSE as the best model
    best_run_idx = df_summary['NRMSE'].idxmin()
    best_run = df_summary.loc[best_run_idx]
    overall_best_automl = best_run['automl_instance']
    
    print(f"\nSelecting overall best model (optimised for {best_run['optimised metric']}) for detailed analysis and saving.")
    
    # save the best model
    with open(model_save_path, 'wb') as f:
        pickle.dump(overall_best_automl, f, pickle.HIGHEST_PROTOCOL)
    print(f"Overall best model saved to {model_save_path}")
    
    return overall_best_automl.model.estimator


def calculate_attributable_mortality(yield_values, beta_value):
    '''calculates mortality attributable to secondary organic aerosol (SOA) exposure
    
    uses a beta coefficient of 8.9 or lower/upper confidence interval deaths per 100,000 people per unit of SOA from Haval et al. (2021) 
    
    args:
        yield_values: array of SOA yield values
        beta_value: mortality coefficient (deaths per 100k people per unit SOA)
    
    returns:
        array of calculated mortality values
    '''
 # mortality coefficient (deaths per 100k people per unit SOA)

    # TO DO, fix current equation by using correct unit value (mass concentration)
    return yield_values * beta_value


def evaluate_and_plot_model(model, data_bundle):
    '''evaluates model performance and creates visualisation of results
    
    this function:
    1. makes predictions on test data
    2. transforms predictions back to original scale
    3. calculates various performance metrics
    4. creates scatter plot of predictions vs actual values
    
    args:
        model: trained machine learning model
        data_bundle: dictionary containing test data and transformation objects
    '''
    print("\nEvaluating Model Performance and Plotting Result...")
    
    # extract necessary data and transformers from the data bundle
    x_test = data_bundle["x_test"]
    y_test_normalised = data_bundle["y_test"]
    y_test_orig = data_bundle["y_test_orig"]
    power_transformer_y = data_bundle["power_transformer_y"]
    normaliser_y = data_bundle["normaliser_y"]
    y_offset = data_bundle["y_offset"]

    # generate predictions on test data
    preds_transformed = model.predict(x_test)

    # transform predictions back to physical units (original scale)
    preds_transformed_reshaped = preds_transformed.reshape(-1, 1)
    preds_unnormalised = normaliser_y.inverse_transform(preds_transformed_reshaped)
    preds_unpowered = power_transformer_y.inverse_transform(preds_unnormalised)
    preds_physical = np.expm1(preds_unpowered).ravel() - y_offset

    # calculate performance metrics on transformed (normalised) data
    rmse_transformed = np.sqrt(mean_squared_error(y_test_normalised, preds_transformed))
    y_range_norm = y_test_normalised.max() - y_test_normalised.min()
    nrmse = rmse_transformed / y_range_norm if y_range_norm != 0 else 0
    r_value = np.corrcoef(y_test_normalised, preds_transformed)[0, 1]
    accuracy_pct = (1 - nrmse) * 100

    # print performance metrics
    print("\nFinal Evaluation Metrics for Best Model")
    print("\nMetrics on Transformed Data:")
    print(f"    NRMSE: {nrmse:.4f}, Model Accuracy: {accuracy_pct:.2f}%, Pearson Correlation (R): {r_value:.4f}")
    print("\nInterpretable Health Metric (on physical data):")
    
    # create scatter plot of predictions vs actual values
    fig1, ax_pred = plt.subplots(figsize=(7, 7))
    
    # plot individual predictions
    ax_pred.plot(y_test_orig, preds_physical, '+', 
                 label='Individual Data', alpha=0.7)
    
    # add perfect prediction line (y=x)
    ax_pred.plot([y_test_orig.min(), y_test_orig.max()], 
                 [y_test_orig.min(), y_test_orig.max()], 
                 '--', color='red', label='1:1 Line')
    
    # customize plot appearance
    ax_pred.set_xlabel('Actual $\gamma$SOA (Physical Units)', fontsize=14)
    ax_pred.set_ylabel('Predicted $\gamma$SOA (Physical Units)', fontsize=14)
    ax_pred.set_title('Predicted vs. Actual Values (Physical Scale)', fontsize=16)
    ax_pred.legend()
    ax_pred.grid(True, linestyle='--', alpha=0.6)
    
    # display the plot
    plt.show()


def predict_time_series_soa():
    '''generates predictions for time series data using quantile regression models
    
    this function:
    1. loads three quantile models (5th, 50th, 95th percentiles)
    2. processes each pickle file in time_series directory
    3. makes predictions for each time step
    4. aggregates predictions with timestamps
    
    returns:
        dictionary containing arrays for lower, median, and upper predictions with timestamps
    '''
    print("\nStarting Time Series SOA Predictions...")
    
    # load quantile models
    models = {}
    quantiles = [5, 50, 95]
    for q in quantiles:
        model_path = Path("quantile_evaluation") / f"quantile_model_{q}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, 'rb') as f:
            models[q] = pickle.load(f)
    
    # initalise predictions storage
    predictions = {
        'lower': [],        # 5th percentile predictions
        'median': [],       # 50th percentile predictions
        'upper': [],        # 95th percentile predictions
        'timestamps': []    # corresponding time points
    }
    
    # verify time series directory exists
    time_series_dir = Path("time_series")
    if not time_series_dir.exists():
        raise FileNotFoundError("time_series directory not found")
    
    # process each time point sequentially
    for day_dir in sorted(time_series_dir.glob("day_*")):
        for hour_file in sorted(day_dir.glob("hour_*.pickle")):
            try:
                # load time point data
                with open(hour_file, 'rb') as f:
                    data = pickle.load(f)
                
                # extract time information
                day = int(day_dir.name.split('_')[1])
                hour = int(hour_file.stem.split('_')[1])
                timestamp = day * 24 + hour
                
                # reshape 3D data to 2D for transformation
                n_params, lat_dim, lon_dim = data.shape
                data_2d = data.reshape(n_params, -1)  # reshape to (n_params, lat_dim * lon_dim)
                
                # prepare input features
                # Note: We're passing a dummy y for transformation as `apply_transformations`
                # expects both X and y. The y_offset won't affect X normalization here.
                features = apply_transformations(data_2d.T, np.zeros(data_2d.T.shape[0]))[0]
                
                # generate predictions for all quantiles
                pred_5 = models[5].predict(features)
                pred_50 = models[50].predict(features)
                pred_95 = models[95].predict(features)
                
                # reshape predictions back to spatial dimensions
                pred_5 = pred_5.reshape(lat_dim, lon_dim)
                pred_50 = pred_50.reshape(lat_dim, lon_dim)
                pred_95 = pred_95.reshape(lat_dim, lon_dim)
                
                # store results
                predictions['lower'].append(pred_5)
                predictions['median'].append(pred_50)
                predictions['upper'].append(pred_95)
                predictions['timestamps'].append(timestamp)
                
                print(f"Processed: Day {day}, Hour {hour}")
            
            except Exception as error_msg:
                print(f"Error processing file {hour_file}: {str(error_msg)}")
    
    # convert predictions to numpy arrays
    for key in predictions:
        if key != 'timestamps':
            predictions[key] = np.array(predictions[key])
        else:
            predictions[key] = np.array(predictions[key], dtype=int)
    
    print("\nTime Series Predictions Complete!")
    print(f"Processed {len(predictions['timestamps'])} time steps")
    
    return predictions  # dictionary with all predictions and timestamps

def run_quantile_regression(data_bundle):
    '''performs quantile regression to estimate uncertainty in predictions and saves models

    this function:
    1. trains models for different quantiles (5th, 50th, and 95th percentiles)
    2. saves the trained models to evaluation directory
    3. generates predictions for each quantile
    4. transforms predictions back to physical units
    5. creates individual scatter plots for each quantile model
    6. calculates mortality based on the model's uncertainty
    7. creates a combined visualisation of the ML model's prediction intervals

    args:
        data_bundle: dictionary containing training and test data
    '''
    print("\nStarting Quantile Regression Uncertainty Analysis:")

    # extract required data from the bundle
    X_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    x_test = data_bundle["x_test"]
    y_test_orig = data_bundle["y_test_orig"]
    y_test_normalised = data_bundle["y_test"] 
    power_transformer_y = data_bundle["power_transformer_y"]
    normaliser_y = data_bundle["normaliser_y"]
    y_offset = data_bundle["y_offset"]

    # define quantiles for lower bound, median, and upper bound
    quantiles = [0.05, 0.5, 0.95]
    models = []
    
    # train a model for each quantile
    for q in quantiles:
        print(f"\nTraining model for quantile: {q}")
        model = xgboost.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=1000,
            random_state=1
        )
        model.fit(X_train, y_train)
        models.append(model)
        
        # Save the model
        save_dir = Path("quantile_evaluation")
        save_dir.mkdir(exist_ok=True)
        model_path = save_dir / f"quantile_model_{int(q*100)}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {int(q*100)}th percentile model to {model_path}")
        
    # generate predictions for each quantile
    preds_transformed = []
    for model in models:
        preds_transformed.append(model.predict(x_test))

    # transform predictions back to physical units
    preds_physical = []
    for p_transformed in preds_transformed:
        p_reshaped = p_transformed.reshape(-1, 1)
        p_unnormalised = normaliser_y.inverse_transform(p_reshaped)
        p_unpowered = power_transformer_y.inverse_transform(p_unnormalised)
        p_physical = np.expm1(p_unpowered).ravel() - y_offset
        preds_physical.append(p_physical)
        
    # unpack predictions for different quantiles
    preds_lower, preds_median, preds_upper = preds_physical

    # debugging number of nans in predictions
    # print(np.isnan(preds_lower).sum())

    print("\n--- Individual Quantile Model Evaluation ---")
    for i in range(len(quantiles)):
        q = quantiles[i]
        p_physical = preds_physical[i]
        p_transformed = preds_transformed[i]
        
        rmse_transformed = np.sqrt(mean_squared_error(y_test_normalised, p_transformed))
        y_range_norm = y_test_normalised.max() - y_test_normalised.min()
        nrmse = rmse_transformed / y_range_norm if y_range_norm != 0 else 0
        r_value = np.corrcoef(y_test_normalised, p_transformed)[0, 1]
        
        print(f"\nMetrics for {int(q*100)}th Percentile Model (on transformed data):")
        print(f"    NRMSE: {nrmse:.4f}, Pearson Correlation (R): {r_value:.4f}")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(y_test_orig, p_physical, '+', label=f'Q={q} Predictions', alpha=0.7)
        ax.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 
                 '--', color='red', label='1:1 Line')
        ax.set_xlabel('Actual $\gamma$SOA (Physical Units)', fontsize=14)
        ax.set_ylabel('Predicted $\gamma$SOA (Physical Units)', fontsize=14)
        ax.set_title(f'Predicted vs. Actual for {int(q*100)}th Percentile Model', fontsize=16)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.show()


    print("Attributable Mortality Uncertainty")

    # confidence interval for mortality rate as a result of SOA exposure by Haval et al. (2021). Deaths per 100,000 people per microgram per cubic metre of SOA
    BETA_BEST = 8.9
    BETA_LOWER = 6.0
    BETA_UPPER = 12.0

    # mortality rate calculation for all three predictions using the single best Beta of 8.9
    am_pred_median = calculate_attributable_mortality(preds_median, beta_value=BETA_BEST)
    am_pred_lower = calculate_attributable_mortality(preds_lower, beta_value=BETA_LOWER)
    am_pred_upper = calculate_attributable_mortality(preds_upper, beta_value=BETA_UPPER)

    print(f"\nAttributable Mortality (deaths per 100k people):")
    # using np.nanmean to calculate the mean, ignoring any NaNs
    print(f"    - Best Estimate (Mean): {np.nanmean(am_pred_median):.4f}")
    print(f"    - Lower Bound from ML (Mean): {np.nanmean(am_pred_lower):.4f}")
    print(f"    - Upper Bound from ML (Mean): {np.nanmean(am_pred_upper):.4f}")


    print("\n--- Demonstration of Variable Prediction Intervals ---")
    print("Showing results for 5 random test samples:")
    # ensures random indices are only chosen from valid (non-NaN) predictions
    valid_indices = np.where(~np.isnan(preds_lower))[0]
    random_indices = np.random.choice(valid_indices, 5, replace=False)
    for i in random_indices:
        actual = y_test_orig[i]
        lower_bound = preds_lower[i]
        median_pred = preds_median[i]
        upper_bound = preds_upper[i]
        interval_width = upper_bound - lower_bound
        print(f"\nSample {i}:")
        print(f"    - Actual γSOA: {actual:.4f}")
        print(f"    - Predicted γSOA (Median): {median_pred:.4f}")
        print(f"    - γSOA 90% Prediction Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"    - γSOA Uncertainty (Interval Width): {interval_width:.4f}")

    # prediction interval visualisation of ML model uncertainty only
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # ensure we only plot non-NaN values for a clean graph
    valid_median_indices = np.where(~np.isnan(preds_median))[0]
    # sort based on the valid median predictions
    sorted_indices = valid_median_indices[np.argsort(preds_median[valid_median_indices])]
    
    x_axis = np.arange(len(sorted_indices))
    
    ax.plot(x_axis, preds_median[sorted_indices], color='blue', label='Median Prediction (50th Percentile)')
    ax.fill_between(x_axis, preds_lower[sorted_indices], preds_upper[sorted_indices], 
                     color='skyblue', alpha=0.4, label='Prediction Interval (5th-95th Percentile)')
    ax.set_xlabel('Test Samples (Sorted by Median Prediction)', fontsize=14)
    ax.set_ylabel('$\gamma$SOA (Physical Units)', fontsize=14)
    ax.set_title('Quantile Regression Prediction Interval for γSOA', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\nStarting Quantile Regression Uncertainty Analysis:")
    
    # extract required data from the bundle
    X_train = data_bundle["X_train"]
    y_train = data_bundle["y_train"]
    x_test = data_bundle["x_test"]
    y_test_orig = data_bundle["y_test_orig"]
    power_transformer_y = data_bundle["power_transformer_y"]
    normaliser_y = data_bundle["normaliser_y"]
    y_offset = data_bundle["y_offset"]
    
    # define quantiles for lower bound, median, and upper bound
    quantiles = [0.05, 0.5, 0.95]  # 5th, 50th, and 95th percentiles
    models = []
    
    # train a model for each quantile
    for q in quantiles:
        print(f"\nTraining model for quantile: {q}")
        model = xgboost.XGBRegressor(
            objective='reg:quantileerror',      # use quantile regression
            alpha=q,                            # specify which quantile to predict
            n_estimators=1000,                  # number of trees in the forest
            random_state=1                      # for reproducibility
        )
        model.fit(X_train, y_train)
        models.append(model)
        
    # generate predictions for each quantile
    preds_transformed = []

    for model in models:
        preds_transformed.append(model.predict(x_test))

    
    # transform predictions back to physical units
    preds_physical = []
    for p_transformed in preds_transformed:
        # reshape predictions for inverse transformation
        p_reshaped = p_transformed.reshape(-1, 1)
        # apply inverse transformations in reverse order
        p_unnormalised = normaliser_y.inverse_transform(p_reshaped)
        p_unpowered = power_transformer_y.inverse_transform(p_unnormalised)
        p_physical = np.expm1(p_unpowered).ravel() - y_offset
        preds_physical.append(p_physical)
        
    # unpack predictions for different quantiles
    preds_lower, preds_median, preds_upper = preds_physical

    # calculate health impact metrics for median predictions
    am_true = calculate_attributable_mortality(y_test_orig, BETA_BEST) 
    am_pred_lower = calculate_attributable_mortality(preds_lower, BETA_LOWER)
    am_pred_median = calculate_attributable_mortality(preds_median, BETA_BEST)
    am_pred_upper = calculate_attributable_mortality(preds_upper, BETA_UPPER)

    mae_am_median = mean_absolute_error(am_true, am_pred_median)
    
    print("\nQuantile Regression Evaluation:")
    print(f"    Median Attributable Mortality Error (MAE-AM): {mae_am_median:.4f} deaths per 100k people")
    print(f"    Lower Bound Attributable Mortality: Min={np.nanmin(am_pred_lower):.4f}, Max={np.nanmax(am_pred_lower):.4f}")
    print(f"    Upper Bound Attributable Mortality: Min={np.nanmin(am_pred_upper):.4f}, Max={np.nanmax(am_pred_upper):.4f}")

    am_interval_width = am_pred_upper - am_pred_lower
    print(f"    Mean Attributable Mortality Prediction Interval Width: {np.nanmean(am_interval_width):.4f} deaths per 100k people")

    # create visualisation of prediction intervals
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # sort predictions for better visualisation
    sorted_indices = np.argsort(preds_median)
    x_axis = np.arange(len(y_test_orig))
    
    # plot median predictions
    ax.plot(x_axis, preds_median[sorted_indices], 
            color='blue', label='Median Prediction (50th Percentile)')
    
    # add shaded area for prediction interval
    ax.fill_between(x_axis, 
                    preds_lower[sorted_indices], 
                    preds_upper[sorted_indices], 
                    color='skyblue', alpha=0.4, 
                    label='Prediction Interval (5th-95th Percentile)')
    
    # customize plot appearance
    ax.set_xlabel('Test Samples (Sorted by Median Prediction)', fontsize=14)
    ax.set_ylabel('$\gamma$SOA (Physical Units)', fontsize=14)
    ax.set_title('Quantile Regression Prediction Interval', fontsize=16)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # adjust layout and display
    plt.tight_layout()
    plt.show()
    

def main():
    '''provides an interactive menu for running different model training and evaluation workflows
    
    menu options:
    1: train and evaluate new AutoML model
    2: quantile regression analysis only
    3: load and evaluate previously trained model
    4: partition parameters into time series
    5: run predictions on time series data
    6: quit the program
    '''
    while True:
        # display menu options
        print("\n" + "="*50)
        print("                     MENU")
        print("-"*50)
        print("1: Train new AutoML model, evaluate and plot results.")
        print("2: Run Quantile Regression uncertainty analysis only.")
        print("3: Load last saved best AutoML model and show its plots.")
        print("4: Partition parameters into time series.")
        print("5: Run predictions on time series data.")
        print("6: Quit.")
        print("="*50)
        
        try:
            # get user choice
            choice = input("Enter your choice [1-6]: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if choice == '1':
            # option 1: automl model training and evaluation
            # load and prepare the data
            data = load_and_prepare_data(path_to_res)
            # train the model using AutoML
            best_model = run_automl_training(data)
            # evaluate and visualize results
            evaluate_and_plot_model(best_model, data)
                
        elif choice == '2':
            # option 2: quantile regression only
            data = load_and_prepare_data(path_to_res)
            run_quantile_regression(data)

        elif choice == '3':
            # option 3: load and evaluate saved model
            # check if required files exist
            if not os.path.exists(model_save_path) or not os.path.exists(summary_save_path):
                print(f"\nError: Model file ('{model_save_path}') or summary file ('{summary_save_path}') not found.")
                print("Please run option 1 to train a model and create these files first.")
                print("Note: Files should be in the auto_ml_evaluation directory.")
                continue
            
            # load and prepare the data
            data = load_and_prepare_data(path_to_res)
            
            # load and display optimisation summary
            print(f"\nLoading optimisation summary from {summary_save_path}...")
            df_summary = pd.read_csv(summary_save_path)
            print("Summary of Last AutoML Optimisation Runs")
            print(df_summary)
            plot_automl_summary_chart(df_summary)

            # load and evaluate the best model
            print(f"\nLoading model from {model_save_path}...")
            with open(model_save_path, 'rb') as f:
                loaded_automl = pickle.load(f)
            best_model = loaded_automl.model.estimator
            evaluate_and_plot_model(best_model, data)

        elif choice == '4':
            print("\nPartitioning parameters into time series...")
            try:
                from picklebatchdivider import main as run_batch_divider
                run_batch_divider()
                print("Parameter partitioning completed successfully.")
            except Exception as error_msg:
                print(f"Error during parameter partitioning: {str(error_msg)}")

        elif choice == '5':
            print("\nRunning predictions on time series data...")
            try:
                # check if required directories exist
                if not Path("time_series").exists():
                    print("Error: time_series directory not found. Please run option 4 first.")
                    continue
                if not Path("quantile_evaluation").exists():
                    print("Error: quantile_evaluation directory not found. Please run quantile regression first.")
                    continue
                
                # run predictions
                predictions = predict_time_series_soa()
                print("\nPredictions completed. Results summary:")
                print(f"Number of time points: {len(predictions['timestamps'])}")
                print(f"Lower bound (5th percentile) shape: {predictions['lower'].shape}")
                print(f"Median prediction shape: {predictions['median'].shape}")
                print(f"Upper bound (95th percentile) shape: {predictions['upper'].shape}")
                
                # create output directory for results
                output_dir = Path("quantile_prediction_results")
                output_dir.mkdir(exist_ok=True)
                
                # save separate pickle files for each quantile
                quantile_files = {
                    'lower': output_dir / "soa_predictions_5th.pkl",
                    'median': output_dir / "soa_predictions_50th.pkl",
                    'upper': output_dir / "soa_predictions_95th.pkl"
                }
                
                for pred_type, file_path in quantile_files.items():
                    with open(file_path, 'wb') as f:
                        pickle.dump({
                            'predictions': predictions[pred_type],
                            'timestamps': predictions['timestamps']
                        }, f)
                    print(f"\nPredictions saved to {file_path}")
                
                # create the histogram plot
                plt.figure(figsize=(10, 7))
                
                # reshape predictions to flatten spatial dimensions
                lower_flat = predictions['lower'].reshape(-1)  # combines all timesteps and spatial points
                median_flat = predictions['median'].reshape(-1)
                upper_flat = predictions['upper'].reshape(-1)

                # determines shared bins for all histograms
                # combines all data to find min and max for consistent binning
                all_preds = np.concatenate([lower_flat, median_flat, upper_flat])
                # filter out NaN values before calculating min/max
                all_preds = all_preds[~np.isnan(all_preds)]
                if all_preds.size > 0:
                    min_val = np.min(all_preds)
                    max_val = np.max(all_preds)
                    bins = np.linspace(min_val, max_val, 60)
                else:
                    print("No valid data to plot histogram.")
                    bins = 10 # default to a small number of bins if no data to avoid error

                if all_preds.size > 0: # only plot if there's valid data
                    plt.hist(lower_flat, bins=bins, color='red', alpha=0.2, histtype='stepfilled', 
                             edgecolor='red', linewidth=1.5, label='5th Percentile')
                    plt.hist(median_flat, bins=bins, color='blue', alpha=0.2, histtype='stepfilled', 
                             edgecolor='blue', linewidth=1.5, label='50th Percentile')
                    plt.hist(upper_flat, bins=bins, color='green', alpha=0.2, histtype='stepfilled', 
                             edgecolor='green', linewidth=1.5, label='95th Percentile')
                
                plt.xlabel('SOA Prediction (Physical Units)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title('Distribution of All SOA Predictions by Quantile', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                # add info about number of points
                n_points = len(lower_flat[~np.isnan(lower_flat)])  # count non-NaN points
                plt.text(0.02, 0.98, f'Total predictions per quantile: {n_points:,}', 
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
                
                # save plot
                plt.savefig(output_dir / 'quantile_predictions_histogram.png')
                plt.close()
                
                print("\nResults have been saved to 'quantile_prediction_results' directory")
                print("- Individual pickle files for each quantile")
                print("- Histogram plot saved as 'quantile_predictions_histogram.png'")
                
            except Exception as error_msg:
                print(f"Error during predictions: {str(error_msg)}")

        elif choice == '6':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")


''' Currently Unused 
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
'''


if __name__ == "__main__":
    # class unused
    class savebestmodel(xgb.callback.TrainingCallback):
        def __init__(self, cvboosters):
            self._cvboosters = cvboosters
            
        def after_training(self, model):
            self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
            return model
    
    main()