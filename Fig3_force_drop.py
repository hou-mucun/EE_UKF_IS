import os
import numpy as np
import matplotlib.pyplot as plt

import evaluation_function as EVF
from Trail import Trail
import models as GIM

'''helf functions'''
def VO2max2AC(VO2max):
    '''convertion from %VO2max to accelerometer count
    :param int VO2max: %VO2max
    '''
    return int((VO2max-1.7228)/0.0135)

data_dir = "simulated_data/Fig2_force_drop"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

'''run simulation'''
trail_info = {'name': 'aft2_60', 'ex_start': 14, 'ex_end': 16, '%VO2max': 60}
trail = Trail(npop = 25, sim_dur = 30, exercise = {'time': [trail_info['ex_start'], trail_info['ex_end']], 'intensity': VO2max2AC(trail_info['%VO2max'])})
trail.create_population()
trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")

iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.65e-5]))**2)
ProcessNoise_enhanced = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.7*1.65e-5]))**2)
ProcessNoise_SBAE = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 3.0*1.65e-5]))**2)
ProcessNoise_SBDE = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 9.0*1.65e-5]))**2)

IS_pop, bas_ISest_pop = trail.simulation(save = 'BG', out = f"{data_dir}/UKF_BG_basal.csv")
np.savetxt(f"{data_dir}/UKF_IS.csv", IS_pop, delimiter=';')
np.savetxt(f"{data_dir}/UKF_ISest_basal.csv", bas_ISest_pop, delimiter=';')
IS_pop, SBAE_ISest_pop = trail.SBAE(iniProcessNoise, ProcessNoise_SBAE, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'],
                                   save = 'BG', out = f"{data_dir}/UKF_BG_SBAE.csv")
np.savetxt(f"{data_dir}/UKF_ISest_SBAE.csv", SBAE_ISest_pop, delimiter=';')
ex_dur = trail_info['ex_start'] - trail_info['ex_start']
IS_pop, SBDE_ISest_pop = trail.SBDE(iniProcessNoise, ProcessNoise_SBDE, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_start'] + ex_dur**2/20,
                                   save = 'BG', out = f"{data_dir}/UKF_BG_SBDE.csv")
np.savetxt(f"{data_dir}/UKF_ISest_SBDE.csv", SBDE_ISest_pop, delimiter=';')

trail.load_models(GIM.fx_mod4, GIM.hx_mod4, 11, 1, "patient_parameters/params_observer_mod4.csv")
iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 1.65e-5]))**2)
ProcessNoise_SBAE = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 3.0*1.65e-5]))**2)
ProcessNoise_SBDE = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 20, 9.0*1.65e-5]))**2)

IS_pop, SBAE_AP_ISest_pop = trail.SBAE(iniProcessNoise, ProcessNoise_SBAE, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_end'],
                                   save = 'BG', out = f"{data_dir}/UKF_BG_SBAE_AP.csv")
np.savetxt(f"{data_dir}/UKF_ISest_SBAE_AP.csv", SBAE_AP_ISest_pop, delimiter=';')
ex_dur = trail_info['ex_start'] - trail_info['ex_start']
IS_pop, SBDE_AP_ISest_pop = trail.SBDE(iniProcessNoise, ProcessNoise_SBDE, t_ex_start = trail_info['ex_start'], t_ex_end = trail_info['ex_start'] + ex_dur**2/20,
                                   save = 'BG', out = f"{data_dir}/UKF_BG_SBDE_AP.csv")
np.savetxt(f"{data_dir}/UKF_ISest_SBDE_AP.csv", SBAE_AP_ISest_pop, delimiter=';')


ID = 14

IS = np.genfromtxt(f"{data_dir}/UKF_IS.csv", delimiter=';')[ID, :]
bas_ISest = np.genfromtxt(f"{data_dir}/UKF_ISest_basal.csv", delimiter=';')[ID, :]
SBAE_ISest = np.genfromtxt(f"{data_dir}/UKF_ISest_SBAE.csv", delimiter=';')[ID, :]
SBDE_ISest = np.genfromtxt(f"{data_dir}/UKF_ISest_SBDE.csv", delimiter=';')[ID, :]
SBAE_AP_ISest = np.genfromtxt(f"{data_dir}/UKF_ISest_SBAE_AP.csv", delimiter=';')[ID, :]
SBDE_AP_ISest = np.genfromtxt(f"{data_dir}/UKF_ISest_SBDE_AP.csv", delimiter=';')[ID, :]

bas_BG = np.genfromtxt(f"{data_dir}/UKF_BG_basal.csv", delimiter=';')[ID, :]
SBAE_BG = np.genfromtxt(f"{data_dir}/UKF_BG_SBAE.csv", delimiter=';')[ID, :]
SBDE_BG = np.genfromtxt(f"{data_dir}/UKF_BG_SBDE.csv", delimiter=';')[ID, :]
SBAE_AP_BG = np.genfromtxt(f"{data_dir}/UKF_BG_SBAE_AP.csv", delimiter=';')[ID, :]
SBDE_AP_BG = np.genfromtxt(f"{data_dir}/UKF_BG_SBDE_AP.csv", delimiter=';')[ID, :]

'''plot'''
dur = len(IS)
ts = 1
steps = int(dur / ts)
t = np.arange(steps) * ts

bas_IS_scale_ratio = np.repeat(100, steps)
enh_IS_scale_ratio = np.repeat(170, steps)
SBAE_IS_scale_ratio = np.repeat(100, steps)
SBAE_IS_scale_ratio[int(trail_info['ex_start']*60/ts) : int(trail_info['ex_end']*60/ts)] = 300
SBDE_IS_scale_ratio = np.repeat(100, steps)
SBDE_IS_scale_ratio[int(trail_info['ex_start']*60/ts) : int((trail_info['ex_start']+0.2)*60/ts)] = 900

fs = 9
lw = 1.25
xpos = np.linspace(0, 1440, 5)
colors = ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']

fig, ax = plt.subplots(2, 1, sharex= False, figsize=(6.4, 9.6))

ax[0].plot(t, IS, linewidth = lw, color = 'k', alpha = 0.8, label = 'true IS')
ax[0].plot(t, bas_ISest, linewidth = lw, color = colors[0], alpha = 0.8, label = 'basal')
ax[0].plot(t, SBAE_ISest, linewidth = lw, color = colors[1], alpha = 0.8, label = 'SBAE')
ax[0].plot(t, SBAE_AP_ISest, linewidth = lw, color = colors[2], alpha = 0.8, label = 'SBAE_AP')
ax[0].plot(t, SBDE_ISest, linewidth = lw, color = colors[3], alpha = 0.8, label = 'SBDE')
ax[0].plot(t, SBDE_AP_ISest, linewidth = lw, color = colors[4], alpha = 0.8, label = 'SBDE_AP')
ax[0].set_ylabel('S$_I$ [ml/$\mu$U/min]', fontsize=fs)
ax[0].set_xticks(xpos)
ax[0].set_xticklabels([str(int(i/60)) for i in xpos])
ax[0].xaxis.set_tick_params(labelsize=fs)
ax[0].set_xlim(t[0]-5, t[-1]+5)
ymin, ymax = ax[0].get_ylim()
ax[0].vlines([840, 960], ymin-0.0001, ymax+0.0002, color='k')
ax[0].set_ylim(ymin-0.0001, ymax+0.0002)
ax[0].legend()
ax[0].grid()

ax[1].plot(t, bas_BG, linewidth = lw, color = colors[0], alpha = 0.5, label = 'basal strategy')
ax[1].plot(t, SBAE_BG, linewidth = lw, color = colors[1], alpha = 0.5, label = 'SBAE strategy')
ax[1].plot(t, SBAE_AP_BG, linewidth = lw, color = colors[2], alpha = 0.5, label = 'SBAE_AP strategy')
ax[1].plot(t, SBDE_BG, linewidth = lw, color = colors[3], alpha = 0.5, label = 'SBDE strategy')
ax[1].plot(t, SBDE_AP_BG, linewidth = lw, color = colors[4], alpha = 0.5, label = 'SBDE_AP strategy')
ax[1].set_ylabel('G [mg/dl]', fontsize=fs)
ax[1].set_xticks(xpos)
ax[1].set_xticklabels([str(int(i/60)) for i in xpos])
ax[1].xaxis.set_tick_params(labelsize=fs)
ax[1].set_xlim(t[0]-5, t[-1]+5)
ymin, ymax = ax[1].get_ylim()
ax[1].vlines([840, 960], ymin-0.0001, ymax+0.0002, color='k')
ax[1].set_ylim(ymin-0.0001, ymax+0.0002)
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig('plot/Fig2_force_drop.png')
plt.show()