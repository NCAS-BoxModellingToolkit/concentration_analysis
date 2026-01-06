<<<<<<< Updated upstream
'''script to quantify the relative contributions of sources and sinks to total PM2.5 mass concentration for multiple households'''
=======
'''script to quantify the relative contributions of sources and '''
'''sinks to total PM2.5 mass concentration for multiple '''
'''households'''
>>>>>>> Stashed changes

# import dependencies
import numpy as np
import sys
import scipy.constants as si
import matplotlib.pyplot as plt
import os
import re
<<<<<<< Updated upstream

# user-defined variables start --------------------------------

# start of paths
base_path = 'C:/Users/user'

# set list of path to results
res_path_sets = [
    [  # household 1
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_complete'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_noegress'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_noingress'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nopartdep'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nobath'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nocls'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nofrag'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nofry'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nopcp'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_nosmo'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_June_wkday_busy_household_novac')
    ],
    [  # household 2
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_August_wkday_combustion_household_complete'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_August_wkday_combustion_household_noegress'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_August_wkday_combustion_household_noingress'),
        str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output/Bradford_August_wkday_combustion_household_nopartdep')
    ]
]

# set corresponding (to path to results) list of plot labels
label_sets = [
    ['complete', 'egress', 'ingress', 'particle deposition', 'showering', 'cleaning spray', 'air freshener', 'frying', 'personal care products', 'smoking', 'vacuuming'],  # household 1
    ['complete', 'egress', 'ingress', 'particle deposition']   # household 2
]

# name of the baseline simulations (must match the entries in 'labels' above)
baseline_labels = ['complete', 'complete']  # household 1, household 2

# path to PyCHAM
PyCHAM_path = 'C:/Users/user/AppData/Local/PyCHAM-5.2.10/PyCHAM-5.2.10/PyCHAM'

# path to save output to
save_path = str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output')
=======
import platform
import pickle

# user-defined variables start --------------------------------

# set base path, depending on operating system
if 'Darwin' in platform.system() or 'Linux' in platform.system():
	base_path = str('/Users/user/Library/CloudStorage/' +
		'OneDrive-TheUniversityofManchester/INGENIOUS/' +
    		'Papers/simulated_PM_mass_versus_observation/')
	PyCHAM_base_path = str('/Users/user/Library/CloudStorage/' +
		'OneDrive-TheUniversityofManchester/')
if 'Win' in platform.system() or 'Linux' in platform.system():
	base_path = str('C:/Users/Psymo/OneDrive - ' +
		'The University of Manchester/INGENIOUS/' +
    		'Papers/simulated_PM_mass_versus_observation/')
	PyCHAM_base_path = str('C:/Users/Psymo/OneDrive - ' +
		'The University of Manchester/')

# path to PyCHAM
PyCHAM_path = str(PyCHAM_base_path +
                  'GitHub/PyCHAM/PyCHAM')

# path to save output to
save_path = str(base_path + 'figures')

# path to where multiple house results saved
multi_house_res_path = str('/Users/user/Library/CloudStorage/' +
						   'OneDrive-TheUniversityofManchester/INGENIOUS/' +
						   'Papers/simulated_PM_mass_versus_observation/' +
						   'programming_scripts/' + 'source_sink_PM2p5_res')

multi_house_res_save_path = str('/Users/user/Library/CloudStorage/' +
						   'OneDrive-TheUniversityofManchester/INGENIOUS/' +
						   'Papers/simulated_PM_mass_versus_observation/' +
						   'figures/' + 'source_sink_PM2p5_res')

# set house code to consider (lo, mo, hi)
house_code = 'lo'

# set list of path to results
if (house_code == 'lo'):
    res_path_sets = [
    [ # mo. house (busy)
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_complete'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_noegress'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_noingress'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_nopartdep'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_nopcp'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_nosho'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_nosmoo'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_noboi'),
        str(base_path + 'PyCHAM_out/Bradford_March_wkday_boiling_household_noocc')
    ]
]
    # set corresponding (to path to results) list of plot labels
    label_sets = [['complete', 'egress', 'ingress', 'particle deposition', 
              'personal care product', 'shower', 'smoke outdoor', 'boil', 
              'occupation']]

# set list of path to results
if (house_code == 'mo'):
    res_path_sets = [
    [ # mo. house (busy)
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_complete'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_noegress'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_noingress'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nopartdep'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nocls'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_noinc'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nofry'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nopcp'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nosho'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_nosmoi'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_novac'),
        str(base_path + 'PyCHAM_out/Bradford_June_wkday_busy_household_noocc')
    ]
]
    # set corresponding (to path to results) list of plot labels
    label_sets = [['complete', 'egress', 'ingress', 'particle deposition', 
              'cleaning spray', 'incense', 'fry', 'personal care product', 
              'shower', 'smoke indoor', 'vacuum', 'occupation']]

if (house_code == 'hi'):
    res_path_sets = [
    [  # household 2
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_complete'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_noairfresh'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_nofcand'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_nocls'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_noegress'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_nofry'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_noingress'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_nopartdep'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_novac'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_nosmoo'),
        str(base_path + 'PyCHAM_out/Bradford_August_wkday_combustion_household_noocc')
		]
]
    # set corresponding (to path to results) list of plot labels
    label_sets = [['complete', 'air freshener', 'candle', 'cleaning spray', 'egress', 
                  'fry', 'ingress', 'particle deposition', 'vacuum', 'smoke outdoor', 
                  'occupation']]
  

# name of the baseline simulations (must match the entries in 'labels' above)
# 'complete', 
baseline_labels = ['complete']  # household 1, household 2
>>>>>>> Stashed changes

# whether to ignore the water mass in the particle phase (1) or not (0)
zero_water_flag = 0

<<<<<<< Updated upstream
# user-defined variables end --------------------------------

# generate unique consistant colour for each source or sink in plots
=======
# whether to save resulting arrays (1) or not (0)
save_arrays_flag = 0

# user-defined variables end --------------------------------

# generate unique consistent colour for each source or sink in plots
>>>>>>> Stashed changes
def base_label(lbl):
    return re.sub(r' \(hh\d+\)$', '', lbl)
all_labels_flat = [base_label(lbl) for sublist in label_sets for lbl in sublist]
all_unique_labels = sorted(set(all_labels_flat))
theme = plt.get_cmap('tab20')
global_label_color_map = {label: theme(i % theme.N) for i, label in enumerate(all_unique_labels)}

# define function
<<<<<<< Updated upstream
def conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, zero_water_flag, baseline_labels):
=======
def conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, 
                        zero_water_flag, baseline_labels, save_arrays_flag):
>>>>>>> Stashed changes
    
    # lists to store stated values
    num_households = len(res_path_sets)
    sources_by_household = {i+1: [] for i in range(num_households)}
    sinks_by_household   = {i+1: [] for i in range(num_households)}

    for household_idx in range(len(res_path_sets)):
        res_path = res_path_sets[household_idx]
        labels = label_sets[household_idx]
        baseline_label = baseline_labels[household_idx]

        # use the global color map for labels to have consistent colours across households
        labels = label_sets[household_idx]
        label_color_map = {label: global_label_color_map[base_label(label)] for label in labels}

        # prepare plots
        fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        
        # ensure PyCHAM can be seen
        sys.path.append(PyCHAM_path)
        import retr_out

        # lists to store stated values
        pm25_mass_list = []
        time_list = []
        
        resi = -1  # simulation index tracker
        
        # loop through simulations
        for res_pathi in res_path:
<<<<<<< Updated upstream
=======
            
>>>>>>> Stashed changes
            resi += 1  # count on simulation results
            # create the self object so that results path is stored
            self = self_def(res_pathi)
            # import results
            try:
                for prog in retr_out.retr_out(self):
                    prog = prog
            except:
                print('Problem importing ', res_pathi)
                import ipdb; ipdb.set_trace()
                continue

            print('now considering simulation: ', res_pathi)

            # get concentrations
            yrec = np.zeros((self.ro_obj.yrec.shape[0], self.ro_obj.yrec.shape[1]))
            yrec[:, :] = self.ro_obj.yrec[:, :]

            # get molar masses of component (g/mol)
            y_MM = np.array((self.ro_obj.comp_MW)).reshape(1, -1)

            # get time (hours) through simulation
            thr = self.ro_obj.thr
            time_mask = thr >= 0  

            # get number of components
            nc = self.ro_obj.nc

<<<<<<< Updated upstream
            # particle-phase concentrations of all components (# molecules/cm^3)
            ppc = yrec[:, self.ro_obj.nc:-self.ro_obj.nc * self.ro_obj.wf]
=======
            # get number of non-particle surfaces
            nnps = self.ro_obj.wf

            # particle-phase concentrations of all components (# molecules/cm^3)
            ppc = yrec[:, nc:-nc*nnps]
>>>>>>> Stashed changes

            # filter to retain only PM2.5 bins
            bin_radii = self.ro_obj.rad # open radius at bin bounds (um)
            bin_diameters = bin_radii * 2 # diameter at bin bounds (um)
            # indices of bins within the PM2.5 size range
            pm25_bin_indices = (bin_diameters < 2.5)
            # prepare to hold individual component concentrations 
            # summed over PM2.5 size bins (molecules/cm^3)
            ppc_pm25 = np.zeros((ppc.shape[0], self.ro_obj.nc))
            
            for it in range(ppc.shape[0]): # loop over times
                # loop over particle size bins
                for ir in range(pm25_bin_indices[it, :].shape[0]):
                    if (pm25_bin_indices[it, ir]) == 1:
                        # sum concentrations over PM2.5-relevant bins (molecules/cm^3)
<<<<<<< Updated upstream
                        ppc_pm25[it, :] += (ppc[it, ir*self.ro_obj.nc:(ir+1)*self.ro_obj.nc])

            # convert # molecules/cm^3 to moles/m^3
            ppc_pm25 = (ppc_pm25 / si.N_A) * 1e6
=======
                        ppc_pm25[it, :] += (ppc[it, ir*nc:(ir+1)*nc])

            # convert # molecules/cm^3 to moles/m^3
            ppc_pm25 = (ppc_pm25/si.N_A) * 1e6
>>>>>>> Stashed changes

            # zero water
            if (zero_water_flag == 1):
                ppc_pm25[:, self.ro_obj.H2O_ind] = 0.

            # convert moles/m^3 to ug/m^3
<<<<<<< Updated upstream
            ppc_pm25 = ppc_pm25 * y_MM * 1e6
=======
            ppc_pm25 = ppc_pm25*y_MM*1e6
>>>>>>> Stashed changes

            # total PM2.5 mass at each timestep (ug/m^3), summing over components
            ppc_total_pm25 = np.sum(ppc_pm25, axis=1)

<<<<<<< Updated upstream
            # plot PM2.5 mass over time
            label = f"{labels[resi]} (hh{household_idx+1})"
=======
            # plot PM2.5 mass over time (ug/m^3)
            label = f"{labels[resi]} (hh{household_idx+1})"
            
>>>>>>> Stashed changes
            ax0.plot(thr[time_mask], ppc_total_pm25[time_mask], label=label, color=label_color_map[labels[resi]])
            # store for comparison later
            pm25_mass_list.append(ppc_total_pm25)
            time_list.append(thr)

        # customise plot appearance - concentration vs time
        if (zero_water_flag == 1):
<<<<<<< Updated upstream
            ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, anhydrous)')
        if (zero_water_flag == 0):
            ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, hydrous)')
=======
            ax0.set_ylabel('PM2.5 Mass Concentration ($\mathrm{\u00B5}$g$\,$m\u207B\u00B3, anhydrous)')
        if (zero_water_flag == 0):
            ax0.set_ylabel('PM2.5 Mass Concentration ($\mathrm{\u00B5}$g$\,$m\u207B\u00B3, hydrous)')
>>>>>>> Stashed changes
        ax0.set_xlabel('Time (hours)')
        ax0.set_title(f'PM2.5 Mass Concentration vs Time (Household {household_idx+1})')
        handles, labels_ = ax0.get_legend_handles_labels()
        clean_labels = [label.replace(' (hh1)', '').replace(' (hh2)', '') for label in labels_]
        ax0.legend(handles, clean_labels, title=f'Scenario Removed from "{baseline_labels[household_idx]}"')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'hh{household_idx+1}_pm25_conc_vs_time.png'))
    
        # define baseline ('complete') simulation 
        baseline_index = labels.index(baseline_label)
        baseline_pm25 = pm25_mass_list[baseline_index]
        baseline_time = time_list[baseline_index]
<<<<<<< Updated upstream
=======
        # sum the baseline PM2.5 over time (ug/m^3)
>>>>>>> Stashed changes
        sum_baseline_pm25 = np.sum(baseline_pm25)

        print(f"\nBaseline: {labels[baseline_index]}")

        # lists to store stated values
        sim_labels = []
        norm_contributions = []
        all_pm_diffs = []

        # calculate total change in PM2.5 mass across all simulations from baseline
        total_change_mass = sum([abs(np.sum(baseline_pm25) - np.sum(pm25_mass_list[i])) for i in range(len(pm25_mass_list)) if i != baseline_index])

        for idx in range(len(pm25_mass_list)):
<<<<<<< Updated upstream
=======
            
>>>>>>> Stashed changes
            if idx == baseline_index:
                continue
            current_pm25 = pm25_mass_list[idx]
            current_time = time_list[idx]
            label = f"{labels[idx]} (hh{household_idx+1})"

            # calculate difference in PM2.5 mass for each simulation from baseline at each timestep
            if np.array_equal(baseline_time, current_time):
                pm_diff = baseline_pm25 - current_pm25
<<<<<<< Updated upstream
                print(f"\nPM2.5 Difference (Baseline '{labels[baseline_index]}' - '{labels[idx]}'): ")
                print(pm_diff)
=======
>>>>>>> Stashed changes

                # calculate total difference in PM2.5 mass for each simulation from baseline over time
                diff_mass = np.trapezoid(pm_diff, x=current_time)

<<<<<<< Updated upstream
                # calculate relative and normalised contributions
                rel_contrib = (diff_mass / sum_baseline_pm25) * 100.0 if sum_baseline_pm25 else 0.0
                rel_contrib_norm = (diff_mass / total_change_mass) * 100.0 if total_change_mass else 0.0

                # store values for printing and plotting
                all_pm_diffs.append((current_time, -1 * pm_diff, label))
=======
                # calculate difference of time-integrated PM2.5 mass relative to the baseline amount
                rel_contrib = (diff_mass / sum_baseline_pm25) * 100.0 if sum_baseline_pm25 else 0.0
                # calculate difference of time-integrated PM2.5 mass relative to the total difference
                rel_contrib_norm = (diff_mass / total_change_mass) * 100.0 if total_change_mass else 0.0

                # store values for printing and plotting
                all_pm_diffs.append((current_time, pm_diff, label))
>>>>>>> Stashed changes
                norm_contributions.append(rel_contrib_norm)
                sim_labels.append(label)

                # total PM2.5 mass for simulation
                sum_current_pm25 = np.sum(current_pm25)

                print(f"\nTotal PM2.5 mass: '{labels[idx]}' = {sum_current_pm25:.2f} μgh/m³")
                print(f"Relative contribution to baseline '{labels[baseline_index]}': {rel_contrib:.2f} %")
                print(f"Normalised contribution to total PM2.5 mass change: {rel_contrib_norm:.2f} %")

<<<<<<< Updated upstream
                # save the PM2.5 mass difference to a CSV file
                filename = f'pm25_diff_baseline_{labels[baseline_index]}_minus_{labels[idx]}.csv'
                np.savetxt(os.path.join(save_path, filename),
=======
                if (save_arrays_flag == 1):
                    # save the PM2.5 mass difference to a CSV file
                    filename = f'pm25_diff_baseline_{labels[baseline_index]}_minus_{labels[idx]}.csv'
                    np.savetxt(os.path.join(save_path, filename),
>>>>>>> Stashed changes
                        np.column_stack((current_time, pm_diff)),
                        delimiter=',',
                        header='Time(hours),PM2.5_Difference(ug/m3)',
                        comments='')
                
        # customise plot appearance - mass difference vs time
        fig_diff_all, ax_diff_all = plt.subplots(figsize=(12, 6))
        for time_arr, pm_diff_arr, label in all_pm_diffs:
            ax_diff_all.plot(time_arr, pm_diff_arr, label=label)
<<<<<<< Updated upstream
=======
            
>>>>>>> Stashed changes
        ax_diff_all.set_title(f'PM2.5 Mass Difference vs Time (Household {household_idx+1})')
        ax_diff_all.set_xlabel('Time (hours)')
        ax_diff_all.set_ylabel('PM2.5 Difference vs Baseline (μg/m³)')
        ax_diff_all.axhline(0, color='black', linestyle='--')
        handles, labels_ = ax_diff_all.get_legend_handles_labels()
        clean_labels = [label.replace(' (hh1)', '').replace(' (hh2)', '') for label in labels_]
        ax_diff_all.legend(handles, clean_labels, title='Scenario Removed')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'hh{household_idx+1}_pm25_diff_vs_time.png'))
        plt.show()

        # separate into sources/sinks per household
        sources = [(label, val) for label, val in zip(sim_labels, norm_contributions) if val > 0]
        sinks = [(label, -val) for label, val in zip(sim_labels, norm_contributions) if val < 0]
        total_sources = sum([v for _, v in sources])
        total_sinks = sum([v for _, v in sinks])

        # normalise
<<<<<<< Updated upstream
        if total_sources > 0:
=======
        if (total_sources > 0):
>>>>>>> Stashed changes
            sources_norm = [(lab, (val / total_sources) * 100) for lab, val in sources]
            sources_by_household[household_idx + 1].extend(sources_norm)

        if total_sinks > 0:
<<<<<<< Updated upstream
            sinks_norm = [(lab, (val / total_sinks) * 100) for lab, val in sinks]
=======
            sinks_norm = [(lab, (-val / total_sinks) * 100) for lab, val in sinks]
>>>>>>> Stashed changes
            sinks_by_household[household_idx + 1].extend(sinks_norm)

    from collections import defaultdict

    def plot_stacked_by_household(data_dict, title, filename):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = list(range(len(data_dict)))  # household positions
        width = 0.6
        bottom = [0, 0]
        theme = plt.get_cmap('tab20')

        # collect all unique labels
        all_labels = sorted(set(
            label for entries in data_dict.values() for label, _ in entries))

        # build lookup
        value_map = {label: [0, 0] for label in all_labels}
        for hh_idx in range(1, len(data_dict) + 1):
            for label, val in data_dict[hh_idx]:
                value_map[label][hh_idx - 1] += val  # accumulate if repeated

        # stack largest contribution on bottom
        label_totals = {label: sum(vals) for label, vals in value_map.items()}
        sorted_labels = sorted(label_totals, key=label_totals.get, reverse=True)

        # plot each label as stacked bar
        bottoms = [0] * len(data_dict)

        def base_label(lbl):
            return re.sub(r' \(hh\d+\)$', '', lbl)

<<<<<<< Updated upstream
=======
        # prepare to hold sorted values and sorted labels
        pickle_res = []
        pickle_name = []

>>>>>>> Stashed changes
        for label in sorted_labels:
            base_lbl = base_label(label)
            ax.bar(x, value_map[label], bottom=bottoms, color=global_label_color_map[base_lbl], label=base_lbl)
            bottoms = [b + v for b, v in zip(bottoms, value_map[label])]
<<<<<<< Updated upstream

        # customise plot appearance - sources/sinks bar chart
        ax.set_xticks(x)
        ax.set_xticklabels([f'Household {i+1}' for i in range(len(data_dict))])
        ax.set_ylabel('Relative Contribution (%)')
=======
            
            pickle_res.append(value_map[label][0])
            pickle_name.append(base_lbl)
        
        # customise plot appearance - sources/sinks bar chart
        ax.set_xticks(x)
        ax.set_xticklabels([f'Household {i+1}' for i in range(len(data_dict))])
        ax.set_ylabel('Relative contribution (%)')
>>>>>>> Stashed changes
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict()
        for handle, label in zip(handles, labels):
            if label not in by_label:
                by_label[label] = handle
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.show()
<<<<<<< Updated upstream

    plot_stacked_by_household(sources_by_household, 
                              'Source Contribution Distribution', 
                              'all_hh_source_relative_contribution.png')
    
    plot_stacked_by_household(sinks_by_household, 
                              'Sink Contribution Distribution', 
                              'all_hh_sink_relative_contribution.png')
=======
        
        # another plot with bars for individual sources/sinks
        fig, ax = plt.subplots(figsize=(7, 6))

        # list the sorted tick labels (w/o hh labelling)
        tick_labels = []
        # list x-axis location of sorted bars
        xloc = []

        x = -1 # keep count on activity
        for labeli in sorted_labels:
            x += 1 # keep count on activity
            base_lbl = base_label(labeli)
            ax.bar(x, value_map[labeli], color=global_label_color_map[base_lbl])

            xloc.append(x)
            tick_labels.append(base_lbl.replace(' ', '\n'))

        ax.set_xticks(xloc, labels=tick_labels)

        # format plot
        ax.yaxis.set_tick_params(labelsize=18, direction = 'in', which='both')
        ax.xaxis.set_tick_params(labelsize=18, direction = 'in', which='both', rotation=90)

        ax.set_ylabel(str(r'Relative contribution to [PM$\mathrm{_{2.5}}$] (%)'), 
		    fontsize = 18)
        
        # ensure tick labels fit on presented window
        plt.tight_layout()
     
        if 'egress' in tick_labels:
            plt.savefig('/Users/user/Library/CloudStorage/OneDrive-TheUniversityofManchester/INGENIOUS/Meetings/UKAC2025/rel_sink_contribution_combustion.pdf', transparent=True)
        if 'egress' not in tick_labels:
            plt.savefig('/Users/user/Library/CloudStorage/OneDrive-TheUniversityofManchester/INGENIOUS/Meetings/UKAC2025/rel_source_contribution_combustion.pdf', transparent=True)


        plt.show()

        return(pickle_res, pickle_name)

    [source_res, source_name] = plot_stacked_by_household(sources_by_household, 
                              'Source Contribution Distribution', 
                              'all_hh_source_relative_contribution.png')
    
    [sink_res, sink_name] = plot_stacked_by_household(sinks_by_household, 
                              'Sink Contribution Distribution', 
                              'all_hh_sink_relative_contribution.png')
    
    # check whether pickle file already exists
    if (os.path.exists(str(multi_house_res_path + '.pickle')) == 0):
        # if not already existing, then create
        with open(str(multi_house_res_path + '.pickle'), 'wb') as pk:
            res = {'lo(source_name, source_res, sink_name, sink_res): ': [[], [], [], []], 
                'mo(source_name, source_res, sink_name, sink_res): ': [[], [], [], []], 
                'hi(source_name, source_res, sink_name, sink_res): ': [[], [], [], []]}
            pickle.dump(res, pk)

    # get existing results
    with open(str(multi_house_res_path + '.pickle'), 'rb') as pk:
        res = pickle.load(pk)
        pk.close()
    
    res[str(house_code + '(source_name, source_res, sink_name, sink_res): ')] = [
        source_name, source_res, sink_name, sink_res]

    # store updated results
    with open(str(multi_house_res_path + '.pickle'), 'wb') as pk: 
        pickle.dump(res, pk) # pickle
        pk.close() # close

    # reference list of activities
    act_ref_list = ['fry', 'air freshener', 'vacuum', 'cleaning spray', 
				 'candle', 'smoke outdoor', 'personal care product', 
				 'boil', 'shower', 'incense', 'smoke indoor', 
                 'ingress', 'egress', 'particle deposition', 'occupation', 'smoke outdoor']
	
    act_colours = ['lightcoral', 'yellow', 'peru', 'cadetblue', 
				 'slategray', 'cyan', 'greenyellow', 
				 'lightcyan', 'blueviolet', 'red', 'deepskyblue', 
                 'goldenrod', 'teal', 'lightpink', 'darkslategray', 'maroon']

    # plot stacked bar charts of source and sink
    # relative contributions per household
    fig, (ax0) = plt.subplots(1, 1, figsize=(9, 5))
	
    hi = 0 # count on dictionary entries (i.e. households)
    # track labels
    labtrak = []
    for di in res:
        for ri in range(2): # loop through source and sink
            # loop through individual sources/sink
            for sii in range(len(res[di][1+2*ri])):
                # check on label occurrence
                if res[di][0+2*ri][sii] not in labtrak:
                    labtrak.append(res[di][0+2*ri][sii])
                    lab_now = res[di][0+2*ri][sii]
                else:
                    lab_now = None
                ax0.bar([ri+(hi*3)], height=[res[di][1+2*ri][sii]], 
                    bottom = sum(res[di][1+2*ri][0:sii]), label=lab_now,
                    width=0.8, 
                    facecolor=act_colours[act_ref_list.index(res[di][0+2*ri][sii])], 
                    edgecolor='k')
                
        hi += 1 # count on dictionary entries (i.e. households)
        
	
    ax0.set_xticks([0, 1, 3, 4, 6, 7], ['lo. sou.', 'lo. sin.', 'mo. sou.', 
        'mo. sin.', 'hi. sou.', 'hi. sin.'])

	# format plot
    ax0.yaxis.set_tick_params(labelsize=18, direction = 'in', which='both')
    ax0.xaxis.set_tick_params(labelsize=18, direction = 'in', which='both', rotation=25)

    ax0.set_ylabel(str('Relative mass contribution\nto ' + '$\mathrm{[PM_{2.5}]}$' + ' (%)'), 
        fontsize = 18)
    
    ax0.set_xlim(left=-0.75, right=11.6)

    ax0.legend(fontsize='12', framealpha=0., loc='upper right')
    plt.tight_layout()
    plt.savefig(str(multi_house_res_save_path + '.pdf'), transparent=True)
    plt.show()
>>>>>>> Stashed changes

# function to setup self
def self_def(dir_path_value):
    class testobj(object):
        pass
    self = testobj()
    self.dir_path = dir_path_value
    return self

# call function
<<<<<<< Updated upstream
conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, zero_water_flag, baseline_labels)
=======
conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, zero_water_flag, 
                baseline_labels, save_arrays_flag)
>>>>>>> Stashed changes
