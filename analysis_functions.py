import numpy as np

# functions give out results for individual patients and population summary

# blood glucose time in range
def computeTIR(BG, lowerBG=80, upperBG=140):

    npop = len(BG)
    dur = len(BG[0])

    TIR = np.zeros(npop)

    for i in range(npop):
        BGpat = BG[i, :]
        TIRpat = len(BGpat[(BGpat >= lowerBG) & (BGpat <= upperBG)]) / dur
        TIR[i] = TIRpat

    TIR_summary = [np.median(TIR), np.percentile(TIR, 25, interpolation='midpoint'),
                   np.percentile(TIR, 75, interpolation='midpoint')]

    return TIR, TIR_summary

# time in hypoglycemia (BG<70mg/dl)
def compute_hypo_time(BG, hypo=70):

    npop = len(BG)
    dur = len(BG[0])

    t_hypo = np.zeros(npop)

    for i in range(npop):
        BGpat = BG[i, :]
        t_hypo_pat = len(BGpat[BGpat < hypo]) / dur
        t_hypo[i] = t_hypo_pat

    t_hypo_summary = [np.median(t_hypo), np.percentile(t_hypo, 25, interpolation='midpoint'),
                      np.percentile(t_hypo, 75, interpolation='midpoint')]

    return t_hypo, t_hypo_summary

# low and high blood glucose indices
def computeGI(BG):

    # compute LBGI and HBGI based on BG risk function
    rl = np.zeros(BG.shape)
    rh = np.zeros(BG.shape)
    for row in range(len(BG)):
        for col in range(len(BG[0])):
            if BG[row, col] < 112.5:
                rl[row, col] = 10 * (1.509 * ((np.log(BG[row, col]))**1.084 - 5.381))**2
            elif BG[row, col] >= 112.5:
                rh[row, col] = 10 * (1.509 * ((np.log(BG[row, col]))**1.084 - 5.381))**2
    LBGI = np.mean(rl, axis=1)
    HBGI = np.mean(rh, axis=1)

    GI_summary = [np.median(LBGI), np.percentile(LBGI, 25, interpolation='midpoint'),
                  np.percentile(LBGI, 75, interpolation='midpoint'),
                  np.median(HBGI), np.percentile(HBGI, 25, interpolation='midpoint'),
                  np.percentile(HBGI, 75, interpolation='midpoint')]

    return LBGI, HBGI, GI_summary

# bolus sizes for b/l/d are summarized
def compute_mealbolus(u):
    u_meal = np.zeros((len(u[0]), 3))
    u_meal[:, 0] = np.median(u, axis=0)
    u_meal[:, 1] = np.percentile(u, 25, axis=0)
    u_meal[:, 2] = np.percentile(u, 75, axis=0)
    return u_meal

'''RMSE of blood glucose in mpc period from target level'''
def BG_RMSE(BG_est, BG_target, period = [21, 30]):
    ts = 1
    BG_est = BG_est[int(period[0]*60/ts):int(period[1]*60/ts)]
    RMSE = np.sqrt(np.mean(np.power(BG_est-BG_target, 2)))
    
    return RMSE

def pop_BG_RMSE(pop_BGest, BG_target, period = [21, 30]):
    npop = len(pop_BGest)
    dur = len(pop_BGest[0,:])
    errors = np.zeros(npop)
    for i in range(npop):
        BGest = pop_BGest[i, :]
        RMSE = BG_RMSE(BGest, BG_target)
        errors[i] = RMSE
        
    return errors

def mean_BG_RMSE(pop_BGest, BG_target, period = [21, 30]):
    return np.mean(pop_BG_RMSE(pop_BGest, BG_target, period = [21, 30]))