import os
import numpy as np
import matplotlib.pyplot as plt

from Trail import Trail
import models as GIM
import evaluation_function as EVF

'''helf functions'''
def VO2max2AC(VO2max):
    '''convertion from %VO2max to accelerometer count
    :param int VO2max: %VO2max
    '''
    return int((VO2max-1.7228)/0.0135)

fs = 9
lw = 1.25
xpos = np.linspace(0, 1440, 5)
colors = [plt.cm.PuBu(0.85), plt.cm.inferno(0.6)]

data_dir = "simulated_data/Fig1_evaluation"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

'''run simulation'''
trail_info = {'name': 'aft2_60', 'ex_start': 14, 'ex_end': 16, '%VO2max': 60}
trail = Trail(npop = 25, sim_dur = 30, exercise = {'time': [trail_info['ex_start'], trail_info['ex_end']], 'intensity': VO2max2AC(trail_info['%VO2max'])})
trail.create_population()
trail.load_models(GIM.fx, GIM.hx, 10, 1, "patient_parameters/params_observer.csv")

IS_pop, bas_ISest_pop = trail.simulation()
np.savetxt(f"{data_dir}/UKF_IS.csv", IS_pop, delimiter=';')
np.savetxt(f"{data_dir}/UKF_ISest_basal.csv", bas_ISest_pop, delimiter=';')

IS_pop = np.genfromtxt(f"{data_dir}/UKF_IS.csv", delimiter=';')
bas_ISest_pop = np.genfromtxt(f"{data_dir}/UKF_ISest_basal.csv", delimiter=';')

ID = 14
IS = IS_pop[ID, :]
ISest = bas_ISest_pop[ID, :]

dur = len(IS)
ts = 1
steps = int(dur / ts)
t = np.arange(steps) * ts

ISest_bfx = ISest[int(6*60/ts):int(14*60/ts)]
IS_bfx = IS[int(6*60/ts):int(14*60/ts)]
RMSE_bfx, MAE_bfx, SE_bfx = EVF.IS_errors(IS_bfx, ISest_bfx)
zone_width = SE_bfx*2

fig, ax = plt.subplots()
ax.plot(t, IS, linewidth = lw, color = colors[0], alpha = 1, label = 'true IS')
ax.plot(t, ISest, linewidth = lw, color = colors[1], alpha = 1, label = 'basal strategy')
for i in t[:int(trail_info['ex_start']*60/ts)-1]:
    ax.fill_between([i-1, i+2], IS[i]-zone_width, IS[i]+zone_width, alpha=0.1, color='grey')
ax.set_ylabel('S$_I$ [ml/$\mu$U/min]', fontsize=fs)
ax.set_xticks(xpos)
ax.set_xticklabels([str(int(i/60)) for i in xpos])
ax.xaxis.set_tick_params(labelsize=fs)
ax.set_xlim(t[0]-5, t[-1]+5)
ymin, ymax = ax.get_ylim()
ax.vlines([840, 960], ymin-0.0001, ymax+0.0002, color='k')
ax.set_ylim(ymin-0.0001, ymax+0.0002)
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig('plot/Fig1_evaluation.png')
plt.show()