import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Trail import Trail
import seaborn as sns
import matplotlib.ticker as tkr

import models as GIM
import evaluation_function as EVF

def VO2max2AC(VO2max):
    '''convertion from %VO2max to accelerometer count
    :param int VO2max: %VO2max
    '''
    return int((VO2max-1.7228)/0.0135)

'''plot setting'''
fs = 9
lw = 1.25
colors = [plt.cm.PuBu(0.85), plt.cm.inferno(0.6)]
xpos = np.linspace(0, 1440, 5)
npop = 25

'''Specify scenarios'''
trails_info = [{'name': 'mor1_40', 'ex_start': 10, 'ex_end': 11}, 
                  {'name': 'mor1_60', 'ex_start': 10, 'ex_end': 11}, 
                  {'name': 'mor2_40', 'ex_start': 9, 'ex_end': 11},
                  {'name': 'mor2_60', 'ex_start': 9, 'ex_end': 11},
                  {'name': 'mor3_40', 'ex_start': 10, 'ex_end': 11},
                  {'name': 'mor3_60', 'ex_start': 10, 'ex_end': 11},  
                  {'name': 'mor4_40', 'ex_start': 7, 'ex_end': 11}, 
                  {'name': 'mor4_60', 'ex_start': 7, 'ex_end': 11},
                  {'name': 'aft1_40', 'ex_start': 15, 'ex_end': 16},
                  {'name': 'aft1_60', 'ex_start': 15, 'ex_end': 16},
                  {'name': 'aft2_40', 'ex_start': 14, 'ex_end': 16}, 
                  {'name': 'aft2_60', 'ex_start': 14, 'ex_end': 16},
                  {'name': 'aft3_40', 'ex_start': 14, 'ex_end': 17}, 
                  {'name': 'aft3_60', 'ex_start': 14, 'ex_end': 17},  
                  {'name': 'aft4_40', 'ex_start': 13, 'ex_end': 17}, 
                  {'name': 'aft4_60', 'ex_start': 13, 'ex_end': 17}]

# Specify exercise scenatios. Fix duration to 30 and all others unchanged.
trail_mor1_40 = Trail(npop, sim_dur = 30, exercise = {'time': [10, 11], 'intensity': VO2max2AC(40)})
trail_mor1_60 = Trail(npop, sim_dur = 30, exercise = {'time': [10, 11], 'intensity': VO2max2AC(60)})
trail_mor2_40 = Trail(npop, sim_dur = 30, exercise = {'time': [9, 11], 'intensity': VO2max2AC(40)})
trail_mor2_60 = Trail(npop, sim_dur = 30, exercise = {'time': [9, 11], 'intensity': VO2max2AC(60)})
trail_mor3_40 = Trail(npop, sim_dur = 30, exercise = {'time': [8, 11], 'intensity': VO2max2AC(40)})
trail_mor3_60 = Trail(npop, sim_dur = 30, exercise = {'time': [8, 11], 'intensity': VO2max2AC(60)})
trail_mor4_40 = Trail(npop, sim_dur = 30, exercise = {'time': [7, 11], 'intensity': VO2max2AC(40)})
trail_mor4_60 = Trail(npop, sim_dur = 30, exercise = {'time': [7, 11], 'intensity': VO2max2AC(60)})

trail_aft1_40 = Trail(npop, sim_dur = 30, exercise = {'time': [15, 16], 'intensity': VO2max2AC(40)})
trail_aft1_60 = Trail(npop, sim_dur = 30, exercise = {'time': [15, 16], 'intensity': VO2max2AC(60)})
trail_aft2_40 = Trail(npop, sim_dur = 30, exercise = {'time': [14, 16], 'intensity': VO2max2AC(40)})
trail_aft2_60 = Trail(npop, sim_dur = 30, exercise = {'time': [14, 16], 'intensity': VO2max2AC(60)})
trail_aft3_40 = Trail(npop, sim_dur = 30, exercise = {'time': [14, 17], 'intensity': VO2max2AC(40)})
trail_aft3_60 = Trail(npop, sim_dur = 30, exercise = {'time': [14, 17], 'intensity': VO2max2AC(60)})
trail_aft4_40 = Trail(npop, sim_dur = 30, exercise = {'time': [13, 17], 'intensity': VO2max2AC(40)})
trail_aft4_60 = Trail(npop, sim_dur = 30, exercise = {'time': [13, 17], 'intensity': VO2max2AC(60)})

trails = np.array([trail_mor1_40, trail_mor1_60, trail_mor2_40, trail_mor2_60, trail_mor3_40, trail_mor3_60, trail_mor4_40, trail_mor4_60,
                   trail_aft1_40, trail_aft1_60, trail_aft2_40, trail_aft2_60, trail_aft3_40, trail_aft3_60 , trail_aft4_40, trail_aft4_60])
    
ntrails = len(trails)

# Specify methods
methods = np.array(['basal',            # original process noise Q
                    'enhanced',         # 170% IS Q throughout dur
                    'SBAE',             # 280%+0.1*ex_dur IS Q throughout exercise
                    'SBAE_AP',
                    'SBDE',              # 900% IS Q with quadratic length      
                    'SBDE_AP'])          
nmethods = len(methods)

crit_list = ["RMSE", "MAE", "SE", "MS", "TLCC"]
ncrit = len(crit_list)

data_dir = "simulated_data/Fig2_scenarios"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

def examine_diff_scenes():
    '''
    '''   
    nsteps = ntrails*nmethods
    performances = np.zeros((ntrails, nmethods, len(crit_list)))
    for i, trail in enumerate(trails):
        trail_info = trails_info[i]
        for j, method in enumerate(methods):
            print(f"--- {i*nmethods+j+1}/{nsteps} --- creating trail {trail_info['name']}, with method {method}")
            ISest_path = f"{data_dir}/{trail_info['name']}/UKF_ISest_{method}.csv"
            IS_path = f"{data_dir}/{trail_info['name']}/UKF_IS.csv"
            
            if os.path.exists(IS_path) and os.path.exists(ISest_path):
                ISest_pop = np.genfromtxt(ISest_path, delimiter=';')
                IS_pop = np.genfromtxt(IS_path, delimiter=';')
            else:
                trail.create_population()
                
                if method == 'basal':
                    trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")
                    IS_pop, ISest_pop = trail.simulation(process_noise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.65e-5]))**2), 
                                            measure_noise = np.diag((np.array([15]))**2))
                elif method == 'enhanced':
                    trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")
                    IS_pop, ISest_pop = trail.simulation(process_noise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.7*1.65e-5]))**2), 
                                            measure_noise = np.diag((np.array([15]))**2))
                elif method == 'SBAE':
                    trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")
                    exe_dur = trail_info['ex_end'] - trail_info['ex_start']
                    iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.65e-5]))**2)
                    exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, (2.8+exe_dur*0.1)*1.65e-5]))**2)
                    IS_pop, ISest_pop = trail.SBAE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'])
                elif method == 'SBAE_AP':
                    trail.load_models(GIM.fx_mod4, GIM.hx_mod4, 11, 1, "patient_parameters/params_observer_mod4.csv")
                    exe_dur = trail_info['ex_end'] - trail_info['ex_start']
                    iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 1.65e-5]))**2)
                    exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, (3.8+exe_dur*0.1)*1.65e-5]))**2)
                    IS_pop, ISest_pop = trail.SBAE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'])
                elif method == 'SBDE_AP':
                    trail.load_models(GIM.fx_mod4, GIM.hx_mod4, 11, 1, "patient_parameters/params_observer_mod4.csv")
                    iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 1.65e-5]))**2)
                    exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 9.0*1.65e-5]))**2)
                    IS_pop, ISest_pop = trail.SBDE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_early_end = trail_info['ex_start']+exe_dur**2/15)
                elif method == 'SBDE':
                    trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")
                    iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.65e-5]))**2)
                    exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 9.0*1.65e-5]))**2)
                    exe_dur = trail_info['ex_end'] - trail_info['ex_start']
                    IS_pop, ISest_pop = trail.SBDE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_early_end = trail_info['ex_start']+exe_dur**2/15)
                else:
                    print(f"method {method} not exists")
                
                direc = f"{data_dir}/{trail_info['name']}"
                if not os.path.isdir(direc):
                    os.makedirs(direc)
                np.savetxt(f"{data_dir}/{trail_info['name']}/UKF_ISest_{method}.csv", ISest_pop, delimiter=';')

            pop_perf = EVF.IS_pop_Evaluator(IS_pop, ISest_pop, 3, 30, trail_info['ex_start'], trail_info['ex_end'])
            performances[i, j, :] = pop_perf
        np.savetxt(f"{data_dir}/{trail_info['name']}/UKF_IS.csv", IS_pop, delimiter=';')
    
    for k, criterion in enumerate(crit_list):
        np.savetxt(f'{data_dir}/perf_scenarios_methods_{criterion}.csv', performances[:, :, k], delimiter=';')

def plot_diff_scenes_heatmap():
    
    f,(ax31,ax32) = plt.subplots(1, 2, figsize=(7.2, 4.8))
    ax31.get_shared_y_axes().join(ax32)
    ax31.set_title(f"RMSE")
    ax32.set_title(f"TCLL")

    perf_RMSE = np.genfromtxt(f'{data_dir}/perf_scenarios_methods_RMSE.csv', delimiter=';')
    perf_RMSE_df = pd.DataFrame(perf_RMSE, columns = methods, index=[trail_info['name'] for trail_info in trails_info])

    perf_CCBTL = np.genfromtxt(f'{data_dir}/perf_scenarios_methods_TLCC.csv', delimiter=';')
    perf_CCBTL_df = pd.DataFrame(perf_CCBTL, columns = methods, index=[trail_info['name'] for trail_info in trails_info])

    formatter1 = tkr.ScalarFormatter(useMathText=True)
    formatter1.set_scientific(True)
    formatter1.set_powerlimits((-1, 2))
    formatter2 = tkr.ScalarFormatter(useMathText=True)
    formatter2.set_scientific(True)
    formatter2.set_powerlimits((-1, 2))

    g1 = sns.heatmap(perf_RMSE_df, ax = ax31, cmap = "rocket_r", cbar_kws={"format": formatter1})
    ax31.tick_params(axis = 'x', labelrotation = 45)
    ax31.tick_params(axis = 'y', labelrotation = 0)
    g2 = sns.heatmap(perf_CCBTL_df, ax = ax32, cmap = "rocket_r", cbar_kws={"format": formatter2})
    ax32.tick_params(axis = 'x', labelrotation = 45)
    ax32.tick_params(axis = 'y', labelrotation = 0)

    plt.tight_layout()
    plt.savefig('plot/Fig3_scenarios.png')
    plt.show()

# examine_diff_scenes()
plot_diff_scenes_heatmap()