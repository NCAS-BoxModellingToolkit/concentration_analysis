'''script to quantify the relative contributions of sources and sinks to total PM2.5 mass concentration for multiple households'''

# import dependencies
import numpy as np
import sys
import scipy.constants as si
import matplotlib.pyplot as plt
import os
import re

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

# whether to ignore the water mass in the particle phase (1) or not (0)
zero_water_flag = 0

# user-defined variables end --------------------------------

# generate unique consistant colour for each source or sink in plots
def base_label(lbl):
    return re.sub(r' \(hh\d+\)$', '', lbl)
all_labels_flat = [base_label(lbl) for sublist in label_sets for lbl in sublist]
all_unique_labels = sorted(set(all_labels_flat))
theme = plt.get_cmap('tab20')
global_label_color_map = {label: theme(i % theme.N) for i, label in enumerate(all_unique_labels)}

# define function
def conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, zero_water_flag, baseline_labels):
    
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

            # particle-phase concentrations of all components (# molecules/cm^3)
            ppc = yrec[:, self.ro_obj.nc:-self.ro_obj.nc * self.ro_obj.wf]

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
                        ppc_pm25[it, :] += (ppc[it, ir*self.ro_obj.nc:(ir+1)*self.ro_obj.nc])

            # convert # molecules/cm^3 to moles/m^3
            ppc_pm25 = (ppc_pm25 / si.N_A) * 1e6

            # zero water
            if (zero_water_flag == 1):
                ppc_pm25[:, self.ro_obj.H2O_ind] = 0.

            # convert moles/m^3 to ug/m^3
            ppc_pm25 = ppc_pm25 * y_MM * 1e6

            # total PM2.5 mass at each timestep (ug/m^3), summing over components
            ppc_total_pm25 = np.sum(ppc_pm25, axis=1)

            # plot PM2.5 mass over time
            label = f"{labels[resi]} (hh{household_idx+1})"
            ax0.plot(thr[time_mask], ppc_total_pm25[time_mask], label=label, color=label_color_map[labels[resi]])
            # store for comparison later
            pm25_mass_list.append(ppc_total_pm25)
            time_list.append(thr)

        # customise plot appearance - concentration vs time
        if (zero_water_flag == 1):
            ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, anhydrous)')
        if (zero_water_flag == 0):
            ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, hydrous)')
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
        sum_baseline_pm25 = np.sum(baseline_pm25)

        print(f"\nBaseline: {labels[baseline_index]}")

        # lists to store stated values
        sim_labels = []
        norm_contributions = []
        all_pm_diffs = []

        # calculate total change in PM2.5 mass across all simulations from baseline
        total_change_mass = sum([abs(np.sum(baseline_pm25) - np.sum(pm25_mass_list[i])) for i in range(len(pm25_mass_list)) if i != baseline_index])

        for idx in range(len(pm25_mass_list)):
            if idx == baseline_index:
                continue
            current_pm25 = pm25_mass_list[idx]
            current_time = time_list[idx]
            label = f"{labels[idx]} (hh{household_idx+1})"

            # calculate difference in PM2.5 mass for each simulation from baseline at each timestep
            if np.array_equal(baseline_time, current_time):
                pm_diff = baseline_pm25 - current_pm25
                print(f"\nPM2.5 Difference (Baseline '{labels[baseline_index]}' - '{labels[idx]}'): ")
                print(pm_diff)

                # calculate total difference in PM2.5 mass for each simulation from baseline over time
                diff_mass = np.trapezoid(pm_diff, x=current_time)

                # calculate relative and normalised contributions
                rel_contrib = (diff_mass / sum_baseline_pm25) * 100.0 if sum_baseline_pm25 else 0.0
                rel_contrib_norm = (diff_mass / total_change_mass) * 100.0 if total_change_mass else 0.0

                # store values for printing and plotting
                all_pm_diffs.append((current_time, -1 * pm_diff, label))
                norm_contributions.append(rel_contrib_norm)
                sim_labels.append(label)

                # total PM2.5 mass for simulation
                sum_current_pm25 = np.sum(current_pm25)

                print(f"\nTotal PM2.5 mass: '{labels[idx]}' = {sum_current_pm25:.2f} μgh/m³")
                print(f"Relative contribution to baseline '{labels[baseline_index]}': {rel_contrib:.2f} %")
                print(f"Normalised contribution to total PM2.5 mass change: {rel_contrib_norm:.2f} %")

                # save the PM2.5 mass difference to a CSV file
                filename = f'pm25_diff_baseline_{labels[baseline_index]}_minus_{labels[idx]}.csv'
                np.savetxt(os.path.join(save_path, filename),
                        np.column_stack((current_time, pm_diff)),
                        delimiter=',',
                        header='Time(hours),PM2.5_Difference(ug/m3)',
                        comments='')
                
        # customise plot appearance - mass difference vs time
        fig_diff_all, ax_diff_all = plt.subplots(figsize=(12, 6))
        for time_arr, pm_diff_arr, label in all_pm_diffs:
            ax_diff_all.plot(time_arr, pm_diff_arr, label=label)
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
        if total_sources > 0:
            sources_norm = [(lab, (val / total_sources) * 100) for lab, val in sources]
            sources_by_household[household_idx + 1].extend(sources_norm)

        if total_sinks > 0:
            sinks_norm = [(lab, (val / total_sinks) * 100) for lab, val in sinks]
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

        for label in sorted_labels:
            base_lbl = base_label(label)
            ax.bar(x, value_map[label], bottom=bottoms, color=global_label_color_map[base_lbl], label=base_lbl)
            bottoms = [b + v for b, v in zip(bottoms, value_map[label])]

        # customise plot appearance - sources/sinks bar chart
        ax.set_xticks(x)
        ax.set_xticklabels([f'Household {i+1}' for i in range(len(data_dict))])
        ax.set_ylabel('Relative Contribution (%)')
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

    plot_stacked_by_household(sources_by_household, 
                              'Source Contribution Distribution', 
                              'all_hh_source_relative_contribution.png')
    
    plot_stacked_by_household(sinks_by_household, 
                              'Sink Contribution Distribution', 
                              'all_hh_sink_relative_contribution.png')

# function to setup self
def self_def(dir_path_value):
    class testobj(object):
        pass
    self = testobj()
    self.dir_path = dir_path_value
    return self

# call function
conc_plot_multi(res_path_sets, label_sets, PyCHAM_path, save_path, zero_water_flag, baseline_labels)