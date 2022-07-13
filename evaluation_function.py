'''Evaluation functions'''

import numpy as np
import pandas as pd


'''Root mean residual sum of square'''
def IS_RMSE(IS, ISest):
    RMSE = np.sqrt(np.mean(np.power(IS-ISest, 2)))
    
    return RMSE

'''Mean absolute error'''
def IS_MAE(IS, ISest):
    MAE = np.mean(np.abs(IS-ISest))
    
    return MAE

'''Mean error'''
def IS_ME(IS, ISest):
    ME = np.mean(IS-ISest)
    
    return ME

'''bias-corrected sample variance'''
def IS_SE(IS, ISest):
    error = IS-ISest
    SE = np.sqrt(np.var(error, ddof = 1))     #Standard deviation of the error
    
    return SE

'''RMSE, MAE, SE for a single patient'''
def IS_errors(IS, ISest):
    RMSE = IS_RMSE(IS, ISest)
    MAE = IS_MAE(IS, ISest)
    SE = IS_SE(IS, ISest)
    
    return RMSE, MAE, SE

'''RMSE, MAE, SE for a whole population'''
def IS_pop_errors(IS_pop, ISest_pop):
    npop = len(IS_pop)
    dur = len(IS_pop[0,:])
    perf_pop = np.zeros((npop, 3))          # three error criteria
    for i in range(npop):
        IS = IS_pop[i, :]
        ISest = ISest_pop[i, :]
        RMSE, MAE, SE = IS_errors(IS, ISest)
        perf_pop[i, :] = np.array([RMSE, MAE, SE])
        
    return perf_pop

def IS_mean_errors(IS_pop, ISest_pop, t_from = 3, t_to = 30, ts = 1):
    # IS population overall errors (from 3:00)
    IS_pop_perf = IS_pop_errors(IS_pop[:, int(t_from*60):int(t_to*60)] , ISest_pop[:, int(t_from*60):int(t_to*60)]) 
    IS_mean_perf = np.mean(IS_pop_perf, axis=0)
    
    return IS_mean_perf              # (3,)

'''Calculate the drop-in time of ISest into an acceptable interval. i.e., zone'''
def drop_in_time(IS, ISest, ts):
    TIME_START = 6
    TIME_EX = 14
    MAX_OUT_TIME = 30
    
    ISest_bfx = ISest[int(TIME_START*60/ts):int(TIME_EX*60/ts)]
    IS_bfx = IS[int(TIME_START*60/ts):int(TIME_EX*60/ts)]
    IS_pat = IS[0]
    SE_bfx = IS_SE(IS_bfx, ISest_bfx)
    zone = np.array([IS_pat - 2*SE_bfx, IS_pat + 2*SE_bfx])
    dropin_t = 0
    dropout_t = -60
    for i in range(1, int(TIME_EX*60/ts)):
        if (ISest[i] <= zone[1] and ISest[i-1] > zone[1]) or (ISest[i] >= zone[0] and ISest[i-1] < zone[0]):
            out_time = i - dropout_t
            if out_time > MAX_OUT_TIME:  
                dropin_t = i
        elif (ISest[i] > zone[1] and ISest[i-1] <= zone[1]) or (ISest[i] < zone[0] and ISest[i-1] >= zone[0]):
            dropout_t = i
    else:
        print("IS drops in at {}min, or, {}h {}min".format(dropin_t, dropin_t*ts//60, dropin_t-(dropin_t*ts//60)*60))
        return dropin_t 
    
'''Shift in time axis between the maxima of IS and ISest'''
def IS_MS(IS, ISest, ts):
    IS_f3 = IS[int(3*60/ts):]               # get rid of the drop-in time
    ISest_f3 = ISest[int(3*60/ts):]
    IS_maxima_t = np.argmax(IS_f3)
    ISest_maxima_t = np.argmax(ISest_f3)
    shift = ISest_maxima_t - IS_maxima_t
    
    return shift

'''cross-correlation based time lag'''
def IS_CCBTL(IS, ISest, ex_start = 14, ex_end = 16, ts = 1):
    ex_dur = ex_end - ex_start
    dur = len(IS)
    steps = int(dur / ts)
    IS_pat = IS[0]
    
    IS_atx = pd.Series(IS[int(ex_start*60/ts):], index = np.arange(int(ex_start*60/ts), steps))
    ISest_atx = pd.Series(ISest[int(ex_start*60/ts):], index = np.arange(int(ex_start*60/ts), steps))
    
    correlations = np.zeros(int(ex_dur*60/ts)+1)
    for lag in range(int(ex_dur*60/ts)+1):
        shifted_IS_atx = IS_atx.shift(lag, fill_value=IS_pat)
        corr = ISest_atx.corr(shifted_IS_atx)
        correlations[lag] = corr
    
    return np.argmax(correlations), correlations

'''time lag (MS and CCBTL) for a whole population'''
def IS_pop_timelag(IS_pop, ISest_pop, ex_start = 14, ex_end = 16, ts = 1):
    npop = len(IS_pop)
    dur = len(IS_pop[0,:])
    steps = int(dur / ts)
    timelag_pop = np.zeros((npop, 2))          # three performance criteria
    for i in range(npop):
        IS = IS_pop[i, :]
        ISest = ISest_pop[i, :]
        Max_shift = IS_MS(IS, ISest, ts)
        corr_shift, corr = IS_CCBTL(IS, ISest, ex_start, ex_end, ts)
        timelag_pop[i, :] = np.array([Max_shift, corr_shift])
        
    return timelag_pop

def IS_pop_Evaluator(IS_pop, ISest_pop, t_from = 3, t_to = 30, ex_start = 14, ex_end = 16, ts = 1):
    
    pop_errors = IS_pop_errors(IS_pop[:, int(t_from*60):int(t_to*60)] , ISest_pop[:, int(t_from*60):int(t_to*60)]) 
    mean_errors = np.mean(pop_errors, axis=0)
    
    # IS population time lag
    pop_timelag = IS_pop_timelag(IS_pop, ISest_pop, ex_start, ex_end, ts)
    mean_timelag = np.mean(pop_timelag, axis=0)
    
    pop_perf = np.concatenate((mean_errors, mean_timelag))
    
    return pop_perf              # (5,) 