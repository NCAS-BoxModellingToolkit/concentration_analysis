import numpy as np
import pickle
from pathlib import Path
import gc
import netCDF4 as nc

# path to the directory containing parameter files
path_to_parameters = r"concentration_analysis/parameters"

# path where time series files will be saved
output_directory = r"time_series"

# dictionary mapping parameter names to their file names

# MINI
parameter_files = {
    "APINENE": "APINENE_(molecules_per_cm^3)_mini",
    "BENZENE": "BENZENE_(molecules_per_cm^3)_mini",
    "CH3O2": "CH3O2_(molecules_per_cm^3)_mini",
    "HO2": "HO2_(molecules_per_cm^3)_mini",
    "JNO2": "JNO2_mini",
    "NO": "NO_(molecules_per_cm^3)_mini",
    "NO3": "NO3_(molecules_per_cm^3)_mini",
    "O3": "O3_(molecules_per_cm^3)_mini",
    "OH": "OH_(molecules_per_cm^3)",
    "OtoC": "OtoC_mini",
    "PM2p5": "PM2p5_(ug_per_m^3)_mini",
    "Pressure": "Pressure_(Pa)_mini",
    "Temperature": "Temperature_(K)_mini",
}

# NORMAL SIZE

# parameter_files = {
#     "APINENE": "APINENE_(molecules_per_cm^3)",
#     "BENZENE": "BENZENE_(molecules_per_cm^3)",
#     "CH3O2": "CH3O2_(molecules_per_cm^3)",
#     "HO2": "HO2_(molecules_per_cm^3)",
#     "JNO2": "J(NO2)_(_per_s)",
#     "NO": "NO_(molecules_per_cm^3)",
#     "NO3": "NO3_(molecules_per_cm^3)",
#     "O3": "O3_(molecules_per_cm^3)",
#     "OH": "OH_(molecules_per_cm^3)",
#     "OtoC": "OtoC",
#     "PM2p5": "selvar_SURF_ug_PM25.nc", # may be misspelt. Haven't tested nc
#     "Pressure": "Pressure_(Pa)",
#     "Temperature": "Temperature_(K)",
# }

class ParameterBatchDivider:
    def __init__(self, parameters_dir, output_dir):
        self.parameters_dir = Path(parameters_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_parameter(self, param_config):
        '''
        loads data from a .pickle or .nc file based on the config.
        - for pickle: param_config is a string (filename).
        - for NetCDF: param_config is a tuple (filename, variable_name).
        '''
        if isinstance(param_config, tuple):
            # if nc file
            filename, var_name = param_config
            param_path = self.parameters_dir / filename
            with nc.Dataset(param_path, 'r') as ds:
                data = np.array(ds.variables[var_name][:])
        else:
            # else pickle file
            param_path = self.parameters_dir / str(param_config)
            with open(param_path, 'rb') as fp:
                data = pickle.load(fp)
                
        data = np.squeeze(data)
        return data
    
    def process_hour(self, hour_index, parameters):
        first_param = next(iter(parameters.values()))
        n_params = len(parameters)
        lat_dim = first_param.shape[1]
        lon_dim = first_param.shape[2]
        
        # create combined array for this hour
        hour_data = np.zeros((n_params, lat_dim, lon_dim))
        
        # fill the array with data from each parameter
        for i, (_, param_data) in enumerate(parameters.items()):
            hour_data[i, :, :] = param_data[hour_index, :, :]
            
        return hour_data
    
    def save_hour_data(self, hour_index, hour_data):
        day = hour_index // 24
        hour = hour_index % 24
        
        # create day directory if it doesn't exist
        day_dir = self.output_dir / f"day_{day:03d}"
        day_dir.mkdir(exist_ok=True)
        
        # save the hour data
        filename = day_dir / f"hour_{hour:02d}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(hour_data, f)

def main():
    try:
        # verify input path exists
        if not Path(path_to_parameters).exists():
            raise FileNotFoundError(f"Parameter directory not found: {path_to_parameters}")
            
        print(f"\nStarting parameter batch processing...")
        print(f"Reading parameters from: {path_to_parameters}")
        print(f"Output will be saved to: {output_directory}")
        
        # initialise the batch divider
        divider = ParameterBatchDivider(path_to_parameters, output_directory)
    
        # load all parameters from the configuration
        param_data = {}
        first_shape = None
        
        print("\nLoading parameters...")
        for param_name, param_config in parameter_files.items():
            print(f"Loading {param_name}...")
            data = divider.load_parameter(param_config)
            
            if first_shape is None:
                first_shape = data.shape  # store the shape of the first parameter
            
            param_data[param_name] = data
        
        print("\nAll parameters loaded successfully. Starting hourly processing...")
        
        # process each hour
        total_hours = first_shape[0] 
        
        for hour in range(total_hours):
            if hour % 100 == 0:
                print(f"Processing hour {hour}/{total_hours}...")
            
            try:
                # process this hour
                hour_data = divider.process_hour(hour, param_data)
                
                # save the hour data
                divider.save_hour_data(hour, hour_data)
                
                # force garbage collection after each save
                gc.collect()
                
            except Exception as error_msg:
                print(f"Error processing hour {hour}: {str(error_msg)}")
                raise
        
        print("Processing complete! All hourly files saved in the time_series directory.")
        
    except Exception as error_msg:
        print(f"An error occurred: {str(error_msg)}")
        raise

if __name__ == "__main__":
    main()