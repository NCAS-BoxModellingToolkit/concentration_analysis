'''script to quantify the relative contributions of sources and sinks to total PM2.5 mass concentration for a single household'''

# import dependencies
import numpy as np
import sys
import scipy.constants as si
import matplotlib.pyplot as plt
import os

# user-defined variables start --------------------------------

# start of paths
base_path = 'C:/Users/user'

# set list of path to results
res_path = [
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
]

# set corresponding (to path to results) list of plot labels
labels = [
    'complete', 'egress', 'ingress', 'particle deposition', 'showering',
    'cleaning spray', 'air freshener', 'frying', 'personal care products',
    'smoking', 'vacuuming'
]

# name of the baseline simulation (must match the entry in 'labels' above)
baseline_label = 'complete'

# path to PyCHAM
PyCHAM_path = 'C:/Users/user/AppData/Local/PyCHAM-5.2.10/PyCHAM-5.2.10/PyCHAM'

# path to save output to
save_path = str(base_path + '/Documents/indoor_environment_PyCHAM_setup/Output')

# whether to ignore the water mass in the particle phase (1) or not (0)
zero_water_flag = 0

# user-defined variables end --------------------------------

# define function
def conc_plot(res_path, labels, PyCHAM_path, save_path, zero_water_flag, baseline_label):
    
    # generate unique consistant colour for each simulation in plots
    theme = plt.get_cmap('tab20')
    label_color_map = {label: theme(i % theme.N) for i, label in enumerate(labels)} 

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
        ax0.plot(thr[time_mask], ppc_total_pm25[time_mask], label=labels[resi], color=label_color_map[labels[resi]])

        # store for comparison later
        pm25_mass_list.append(ppc_total_pm25)
        time_list.append(thr)

    # customise plot appearance
    if (zero_water_flag == 1):
        ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, anhydrous)', fontsize=14)
    if (zero_water_flag == 0):
        ax0.set_ylabel('PM2.5 Mass Concentration (μg/m³, hydrous)', fontsize=14)
    ax0.set_xlabel('Time (hours)', fontsize=14)
    ax0.set_title('PM2.5 Mass Concentration over Time', fontsize=14)
    ax0.yaxis.set_tick_params(labelsize=14, direction='in', which='both')
    ax0.xaxis.set_tick_params(labelsize=14, direction='in', which='both')
    ax0.legend(title='Scenario Removed', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pm25_mass_concentration_vs_time.png'))

    # precompute total PM2.5 mass for all simulations
    total_pm25_sums = [np.sum(pm25) for pm25 in pm25_mass_list]  
    # define baseline ('complete') simulation 
    baseline_index = labels.index(baseline_label)
    sum_baseline_pm25 = total_pm25_sums[baseline_index] 
    baseline_pm25 = pm25_mass_list[baseline_index]
    baseline_time = time_list[baseline_index]

    print(f"\nBaseline: {labels[baseline_index]}")

    # lists to store stated values
    total_pm25_mass = []
    relative_contributions = []
    norm_contributions = []
    sim_labels = []
    all_pm_diffs = []

    # calculate total change in PM2.5 mass across all simulations from baseline
    total_change_mass = sum([abs(np.sum(baseline_pm25) - np.sum(pm25_mass_list[idx]))
        for idx in range(len(pm25_mass_list)) if idx != baseline_index
    ])
    for idx in range(len(pm25_mass_list)):
        if idx == baseline_index:
            continue

        current_pm25 = pm25_mass_list[idx]
        current_time = time_list[idx]

        # calculate difference in PM2.5 mass for each simulation from baseline at each timestep
        if np.array_equal(baseline_time, current_time):
            pm_diff = baseline_pm25 - current_pm25
            print(f"\nPM2.5 Difference (Baseline '{labels[baseline_index]}' - '{labels[idx]}'): ")
            print(pm_diff)

            # calculate total difference in PM2.5 mass for each simulation from baseline over time
            sum_current_pm25 = total_pm25_sums[idx]
            current_total_mass = sum_current_pm25
            diff_mass = np.trapezoid(baseline_pm25 - current_pm25, x=current_time)

            # calculate relative and normalised contributions
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_contrib = (diff_mass / sum_baseline_pm25) * 100.0 
                rel_contrib = 0.0 if np.isnan(rel_contrib) or np.isinf(rel_contrib) else rel_contrib

                rel_contrib_norm = (diff_mass / total_change_mass) * 100.0 if total_change_mass != 0 else 0.0
                rel_contrib_norm = 0.0 if np.isnan(rel_contrib_norm) or np.isinf(rel_contrib_norm) else rel_contrib_norm

            # store values for printing and plotting
            total_pm25_mass.append(current_total_mass)
            relative_contributions.append(rel_contrib)
            norm_contributions.append(rel_contrib_norm)
            sim_labels.append(labels[idx])

            print(f"\nTotal PM2.5 mass: '{labels[idx]}' = {current_total_mass:.2f} μgh/m³")
            print(f"Relative contribution to baseline '{labels[baseline_index]}': {rel_contrib:.2f} %")
            print(f"Normalised contribution to total PM2.5 mass change: {rel_contrib_norm:.2f} %")

            # save diff for plotting later
            all_pm_diffs.append((current_time, pm_diff, labels[idx]))

            # save the PM2.5 mass difference to a CSV file
            filename = f'pm25_diff_baseline_{labels[baseline_index]}_minus_{labels[idx]}.csv'
            np.savetxt(os.path.join(save_path, filename),
                       np.column_stack((current_time, pm_diff)),
                       delimiter=',',
                       header='Time(hours),PM2.5_Difference(ug/m3)',
                       comments='')
            
    # plot PM2.5 differences over time
    fig_diff_all, ax_diff_all = plt.subplots(figsize=(12, 6))
    for time_arr, pm_diff_arr, label in all_pm_diffs:
        flipped_diff = -1 * pm_diff_arr
        ax_diff_all.plot(time_arr, flipped_diff, label=label, color=label_color_map[label])

    # customise plot appearance
    ax_diff_all.set_title(
        f'PM2.5 Mass Difference over Time\nRelative to Baseline', fontsize=14)
    ax_diff_all.set_xlabel('Time (hours)', fontsize=12)
    ax_diff_all.set_ylabel('PM2.5 Difference (μg/m³)', fontsize=12)
    ax_diff_all.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_diff_all.grid(True)
    ax_diff_all.legend(title='Scenario Removed', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'pm25_diff_vs_time.png'))
    plt.show()

    # sort simulations by size for bar chart
    sorted_data = sorted(zip(sim_labels, norm_contributions), key=lambda x: abs(x[1]), reverse=True)
    sim_labels, norm_contributions = zip(*sorted_data)

    # plot stacked barchart
    fig, ax = plt.subplots(figsize=(12, 6))

    pos_bottom = 0
    neg_bottom = 0

    for i, val in enumerate(norm_contributions):
        label = sim_labels[i]
        color = plt.cm.tab10(i % 10)

        if val > 0:
            # sources stack upwards from 0
            ax.bar(0, val, bottom=pos_bottom, label=label, color=label_color_map[label])
            pos_bottom += val
        elif val < 0:
            # sinks stack downwards from 0
            ax.bar(0, val, bottom=neg_bottom, label=label, color=label_color_map[label])
            neg_bottom += val

    # customise plot appearance
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_ylabel('Relative Contribution (%)', fontsize=14)
    ax.set_title(
        'Sources and Sinks of Indoor PM2.5 Mass Concentration\nfor a Single Household over a 24-Hour Period',
        fontsize=14,
        wrap=True
    )
    ax.set_xticks([])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'relative_contribution.png'))
    plt.show()

    # sort into sources and sinks
    sources = [(lab, val) for lab, val in zip(sim_labels, norm_contributions) if val > 0]
    sinks = [(lab, -val) for lab, val in zip(sim_labels, norm_contributions) if val < 0]  

    total_sources = sum([v for _, v in sources])
    total_sinks = sum([v for _, v in sinks])

    # normalise to 100% of category (source or sink)
    if total_sources > 0:
        sources_norm = [(lab, (val / total_sources) * 100) for lab, val in sources]
    else:
        sources_norm = []

    if total_sinks > 0:
        sinks_norm = [(lab, (val / total_sinks) * 100) for lab, val in sinks]
    else:
        sinks_norm = []

    # customise source plot appearance 
    if sources_norm:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_src, values_src = zip(*sources_norm)
        x = range(len(labels_src))
        colors_src = [label_color_map[label] for label in labels_src]
        ax.bar(x, values_src, color=colors_src, width=0.8)
        ax.set_ylabel('Relative Contribution (%)', fontsize=14)
        ax.set_title('Source Contribution Distribution', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_src, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'relative_contribution_sources.png'))
        plt.show()

    # customise sink plot appearance
    if sinks_norm:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels_sink, values_sink = zip(*sinks_norm)
        x = range(len(labels_sink))
        colors_sink = [label_color_map[label] for label in labels_sink]
        ax.bar(x, values_sink, color=colors_sink, width=0.8)
        ax.set_ylabel('Relative Contribution (%)', fontsize=14)
        ax.set_title('Sink Contribution Distribution', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_sink, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'relative_contribution_sinks.png'))
        plt.show()

# function to setup self
def self_def(dir_path_value):
    class testobj(object):
        pass
    self = testobj()
    self.dir_path = dir_path_value
    return self

# call function
conc_plot(res_path, labels, PyCHAM_path, save_path, zero_water_flag, baseline_label)
