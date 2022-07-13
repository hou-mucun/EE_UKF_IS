'''Modified from Julia's UKF_IS
Reference:
Deichmann, J., & Kaltenbach, H. M. (2021). 
Estimating insulin sensitivity after exercise using an Unscented Kalman Filter. 
Ifac Papersonline, 54(15), 472-477. doi:10.1016/j.ifacol.2021.10.301
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Trail import Trail
import models as GIM
import analysis_functions as ASF

def VO2max2AC(VO2max):
    '''convertion from %VO2max to accelerometer count
    :param int VO2max: %VO2max
    '''
    return int((VO2max-1.7228)/0.0135)

'''mpc setting'''
period = [0,30]
sample_time = 5
pred_horizon = 14
mpc_param = f"{period[0]}-{period[1]}_{sample_time}_{pred_horizon}"

'''data preparation'''
trail_info = {'name': 'aft4_60', 'ex_start': 13, 'ex_end': 17, '%VO2max': 60}
trail = Trail(npop = 25, sim_dur = 30, exercise = {'time': [trail_info['ex_start'], trail_info['ex_end']], 'intensity': VO2max2AC(trail_info['%VO2max'])})
trail.create_population()
trail.load_models(GIM.fx_mod4, GIM.hx_mod4, 11, 1, "patient_parameters/params_observer_mod4.csv")

method = 'SBAE_AP'

scale_ratio = 4.2
iniProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1, 1.65e-5]))**2)
exeProcessNoise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1, scale_ratio*1.65e-5]))**2)

outpath = f"simulated_data/Fig4_mpc/{trail_info['name']}/{method}"
if not os.path.isdir(outpath):
    os.makedirs(outpath)

pop_BG_pc = np.genfromtxt(f"{outpath}/UKF_BG_pc.csv", delimiter=';')
pop_BG_na = np.genfromtxt(f"{outpath}/UKF_BG_na.csv", delimiter=';')
pop_BG_mpc = np.genfromtxt(f"{outpath}/UKF_BG_{mpc_param}.csv", delimiter=';')

analysis_start = trail_info['ex_start']   # from start of PA

# TIR 80-140mg/dl
TIR_pc, TIRsummary_pc = ASF.computeTIR(pop_BG_pc[:, analysis_start:])
TIR_mpc, TIRsummary_mpc = ASF.computeTIR(pop_BG_mpc[:, analysis_start:])
# TIR 70-180mg/dl
TIR2_pc, TIRsummary2_pc = ASF.computeTIR(pop_BG_pc[:, analysis_start:], 70, 180)
TIR2_mpc, TIRsummary2_mpc = ASF.computeTIR(pop_BG_mpc[:, analysis_start:], 70, 180)
# time < 70mg/dl
thypo_pc, thypo_summary_pc = ASF.compute_hypo_time(pop_BG_pc[:, analysis_start:])
thypo_mpc, thypo_summary_mpc = ASF.compute_hypo_time(pop_BG_mpc[:, analysis_start:])
# LBGI and HBGI
LBGI_pc, HBGI_pc, GI_pc = ASF.computeGI(pop_BG_pc[:, analysis_start:])
LBGI_mpc, HBGI_mpc, GI_mpc = ASF.computeGI(pop_BG_mpc[:, analysis_start:])
# summarize results
results_pc = [str(np.round(TIRsummary_pc[0]*100,1))+' ['+str(np.round(TIRsummary_pc[1]*100,1))+
                    ', ' + str(np.round(TIRsummary_pc[2]*100,1)) +']',
                    str(np.round(TIRsummary2_pc[0]*100,1))+' ['+str(np.round(TIRsummary2_pc[1]*100,1))
                    +', '+str(np.round(TIRsummary2_pc[2]*100,1))+']',
                    str(np.round(thypo_summary_pc[0]*100,1))+' ['+str(np.round(thypo_summary_pc[1]*100,1))
                    +', '+str(np.round(thypo_summary_pc[2]*100,1))+']',
                    str(np.round(GI_pc[0], 2))+' ['+str(np.round(GI_pc[1], 2))
                    +', '+str(np.round(GI_pc[2], 2))+']',
                    str(np.round(GI_pc[3], 2))+' ['+str(np.round(GI_pc[4], 2))
                    +', '+str(np.round(GI_pc[5], 2))+']']
results_mpc = [str(np.round(TIRsummary_mpc[0]*100,1))+' ['+str(np.round(TIRsummary_mpc[1]*100,1))+
                    ', ' + str(np.round(TIRsummary_mpc[2]*100,1)) +']',
                    str(np.round(TIRsummary2_mpc[0]*100,1))+' ['+str(np.round(TIRsummary2_mpc[1]*100,1))
                    +', '+str(np.round(TIRsummary2_mpc[2]*100,1))+']',
                    str(np.round(thypo_summary_mpc[0]*100,1))+' ['+str(np.round(thypo_summary_mpc[1]*100,1))
                    +', '+str(np.round(thypo_summary_mpc[2]*100,1))+']',
                    str(np.round(GI_mpc[0], 2))+' ['+str(np.round(GI_mpc[1], 2))
                    +', '+str(np.round(GI_mpc[2], 2))+']',
                    str(np.round(GI_mpc[3], 2))+' ['+str(np.round(GI_mpc[4], 2))
                    +', '+str(np.round(GI_mpc[5], 2))+']']

# create data frame
results_fromPA = pd.DataFrame(data={'Propertional': results_pc, 'MPC': results_mpc},
                              index=('TIR (80-140mg/dl)', 'TIR (70-180mg/dl)', 'time <70mg/dl', 'LBGI', 'HBGI'))
print('Summary of results from PA start.')
print(results_fromPA)