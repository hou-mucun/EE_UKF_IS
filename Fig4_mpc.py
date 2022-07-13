import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Trail import Trail
import models as GIM

def VO2max2AC(VO2max):
    '''convertion from %VO2max to accelerometer count
    :param int VO2max: %VO2max
    '''
    return int((VO2max-1.7228)/0.0135)

'''plot setting'''
fs = 9
lw = 1.25
color = plt.cm.viridis([0.9, 0.62, 0.1])

'''mpc setting'''
period = [0,30]
sample_time = 5
pred_horizon = 14
mpc_param = f"{period[0]}-{period[1]}_{sample_time}_{pred_horizon}"

trail_infos = [{'name': 'aft2_60', 'ex_start': 14, 'ex_end': 16, '%VO2max': 60}, {'name': 'aft4_60', 'ex_start': 13, 'ex_end': 17, '%VO2max': 60}]
method = 'SBAE_AP'

'''prepare for data'''
for i, trail_info in enumerate(trail_infos):
    trail = Trail(npop = 25, sim_dur = 30, exercise = {'time': [trail_info['ex_start'], trail_info['ex_end']], 'intensity': VO2max2AC(trail_info['%VO2max'])})
    trail.create_population()
    trail.load_models(GIM.fx_mod4, GIM.hx_mod4, 11, 1, "patient_parameters/params_observer_mod4.csv")

    scale_ratio = 4.2
    iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1, 1.65e-5]))**2)
    exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1, scale_ratio*1.65e-5]))**2)

    outpath = f"simulated_data/Fig4_mpc/{trail_info['name']}/{method}"
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    ## BG_pc: BG with proportional control for basal insulin infusion
    IS_pop, ISest_pop = trail.SBAE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'],
                                basal_adjust = True, save = 'BG', out = f"{outpath}/UKF_BG_pc.csv")
    ## BG_na: BG with no adjustment of basal insulin infusion
    IS_pop, ISest_pop = trail.SBAE(iniProcessNoise, exeProcessNoise, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'],
                                basal_adjust = False, save = 'BG', out = f"{outpath}/UKF_BG_na.csv")
    ## BG_mpc: BG with mpc control
    pop_BG, pop_BGest, u_seq = trail.mpc(iniProcessNoise, exeProcessNoise, period = period, sample_time = sample_time, pred_horizon = pred_horizon, BG_target = 110, 
            t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'])
    np.savetxt(f"{outpath}/UKF_BG_{mpc_param}.csv", pop_BG, delimiter=';')
    np.savetxt(f"{outpath}/UKF_u_seq_{mpc_param}.csv", u_seq, delimiter=';')

'''plot'''

ID = 14
xpos = np.linspace(0, 1440, 5)
ts = 1

fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))

for i, trail_info in enumerate(trail_infos):
    outpath = f"simulated_data/Fig4_mpc/{trail_info['name']}/{method}"
    
    pop_BG_pc = np.genfromtxt(f"{outpath}/UKF_BG_pc.csv", delimiter=';')
    pop_BG_na = np.genfromtxt(f"{outpath}/UKF_BG_na.csv", delimiter=';')
    pop_BG_mpc = np.genfromtxt(f"{outpath}/UKF_BG_{mpc_param}.csv", delimiter=';')
    
    npop = len(pop_BG_pc)
    dur = len(pop_BG_pc[0,:])
    steps = int(dur / ts)
    t = np.arange(steps) * ts

    # min, max and mean
    min_BG_na = np.min(pop_BG_na, axis=0)
    max_BG_na = np.max(pop_BG_na, axis=0)
    mean_BG_na = np.mean(pop_BG_na, axis=0)

    min_BG_pc = np.min(pop_BG_pc, axis=0)
    max_BG_pc = np.max(pop_BG_pc, axis=0)
    mean_BG_pc = np.mean(pop_BG_pc, axis=0)

    min_BG_mpc = np.min(pop_BG_mpc, axis=0)
    max_BG_mpc = np.max(pop_BG_mpc, axis=0)
    mean_BG_mpc = np.mean(pop_BG_mpc, axis=0)


    ax[i].fill_between(t, min_BG_na, max_BG_na, alpha=.4, color=color[0])
    ax[i].plot(t, mean_BG_na, linewidth=lw, label="BG_na", color=color[0])
    ax[i].fill_between(t, min_BG_pc, max_BG_pc, alpha=.4, color=color[1])
    ax[i].plot(t, mean_BG_pc, linewidth=lw, label="BG_pc", color=color[1])
    ax[i].fill_between(t, min_BG_mpc, max_BG_mpc, alpha=.4, color=color[2])
    ax[i].plot(t, mean_BG_mpc, linewidth=lw, label="BG_mpc", color=color[2])
    ymin, ymax = ax[i].get_ylim()
    ax[i].fill_between([t[0] - 5, t[-1] + 5], 0, 70, alpha=0.3, color='grey')
    ax[i].fill_between([t[0] - 5, t[-1] + 5], 180, 220, alpha=0.3, color='grey')
    ax[i].vlines([int(trail_info['ex_start']*60/ts), int(trail_info['ex_end']*60/ts)], ymin-5, ymax+5, color='k')
    ax[i].vlines(1320, ymin-5, ymax+5, color='k', linestyles='dashed')
    ax[i].set_xlabel('time [h]', fontsize=fs)
    ax[i].set_xticks(xpos)
    ax[i].set_xticklabels([str(int(i/60)) for i in xpos])
    ax[i].xaxis.set_tick_params(labelsize=fs)
    ax[i].set_xlim(t[0]-5, t[-1]+5)
    ax[i].set_ylim(ymin-5, ymax+5)
    ax[i].set_ylabel('G [mg/dl]', fontsize=fs)
    ax[i].yaxis.set_tick_params(labelsize=fs)
    ax[i].legend(fontsize=fs)
    ax[i].grid()

plt.tight_layout()
plt.savefig('plot/Fig4_mpc.png')
plt.show()