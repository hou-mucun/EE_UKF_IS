# OOP based patient simulation package

import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman.sigma_points import JulierSigmaPoints

import models as GIM
import bolus_calculator as IBC

class Trail:
    
    def __init__(self, npop, sim_dur, ts = 1, BW = 70, Gb = 110, 
                 IS_nom = 2.47e-5/0.015, 
                 m2_nom = 0.0513,
                 meal = {'time': [7, 12, 18], 'D': [60e3, 80e3, 70e3]}, 
                 exercise = {'time': [14, 16], 'intensity': 4317}, 
                 extra_CHO = {'time': 16, 'D': 15e3}, 
                 basal_insulin_time = 22):
        ''' Initialize the simulated trail for insulin glucose dynamics
        
        :param int npop: the size of patient population
        :param int sim_dur: simulation duration [h]
        :param int ts: simulation interval [min]
        :param float BW: body weight set for all patients
        :param float Gb: basal/target glucose [mg/dl]
        :param float IS_nom: nominal insulin sensitivity
        :param float m2_nom: nominal meal digestion m2
        :param dic meal: meal time and mass {'time': [3], 'D': [3]}
        :param dic exercise: exercise time and intensity {'time': int, 'intensity': float}
        :param dic extra_CHO: time and mass for extra CHO {'time': int, 'D': float}
        :param int basal_insulin_time: time for basal insulin adjustment
        '''
        self.npop = npop
        self.dur = int(sim_dur*60)
        self.ts = ts
        self.BW = BW
        self.Gb = Gb
        self.IS_nom = IS_nom
        self.m2_nom = m2_nom
        self.tmeal = meal['time']
        self.Dmeal = meal['D']
        self.tex = exercise['time']
        self.intex = exercise['intensity']
        self.textra_CHO = extra_CHO['time']
        self.Dextra_CHO = extra_CHO['D']
        self.basel_insulin_time = basal_insulin_time
        
        self.patient_char = None
        self.noise = None
        self.m2_pop = None
        self.t_meals = None
        self.t_u_basal = None
        self.AC = None
        self.D = None
        
    
    def create_population(self, p1 = 0.008, EGP0 = 3.469):
        '''create population, prepare patient specific parameters, and define input sequences.
        :param double p1: rate parameter
        :param double EGP0: glucose production rate at zero glucose
        '''
        nmeals = len(self.tmeal)
        Qb = self.Gb*1.289
        
        # standard deviation used to draw parameter values from normal distribution
        sig_m2 = 0.25 * self.m2_nom
        sig_IS = 0.25 * self.IS_nom
        # generate parameter values
        np.random.seed(13)
        m2_params = self.m2_nom + sig_m2 * np.random.randn(self.npop, nmeals)    # new m2 for each meal
        np.random.seed(17)
        IS_pop = self.IS_nom + sig_IS * np.random.randn(self.npop)            # new IS for each patient

        ### compute basal insulin
        Ib_nom = (EGP0 / Qb - p1) / self.IS_nom      # 9.979248
        Ib_pop = (EGP0 / Qb - p1) / IS_pop

        ### patient-specific ICR and CF ###
        ICR0 = 10
        CF0 = 14.5

        ICR = ICR0 * IS_pop / self.IS_nom
        CF = CF0 * IS_pop / self.IS_nom

        ### save in dataframe ###
        p = pd.DataFrame(data=np.zeros((self.npop, 4)), columns={'IS', 'Ib', 'ICR', 'CF'})
        p['IS'] = IS_pop
        p['Ib'] = np.round(Ib_pop, 2)
        p['ICR'] = np.round(ICR, 1)
        p['CF'] = np.round(CF, 1)
        
        ''' generate noise sequence '''

        noise = np.zeros((self.dur, self.npop))

        np.random.seed(100)
        wcc = np.random.randn(self.npop, self.dur) * np.sqrt(11.3)
        w = np.random.randn(self.npop, self.dur) * np.sqrt(14.45)

        for j in range(self.npop):
            cc = np.zeros(self.dur)
            v = np.zeros(self.dur)
            for i in range(self.dur):
                if i == 0:
                    cc[i] = wcc[j, i]
                    v[i] = w[j, i]
                elif i == 1:
                    cc[i] = 1.23 * cc[i-1] + wcc[j, i]
                    v[i] = 1.013 * v[i-1] + w[j, i]
                else:
                    cc[i] = 1.23 * cc[i - 1] - 0.3995 * cc[i-2] + wcc[j, i]
                    v[i] = 1.013 * v[i - 1] - 0.2135 * v[i-2] + w[j, i]
            noise[:, j] = cc + v

        ''' store results '''
        self.patient_char = p
        self.noise = noise
        
        steps = int(self.dur / self.ts)
        D = np.zeros(steps)
        AC = np.zeros(steps)
        
        t_meals = [int(self.tmeal[0]*60/self.ts), int(self.tmeal[1]*60/self.ts), int(self.tmeal[2]*60/self.ts)]
        t_u_basal = int(self.basel_insulin_time*60/self.ts)
        
        for i in range(3): D[t_meals[i]] = self.Dmeal[i]

        m2_pop = np.zeros((self.npop, int(self.dur/self.ts)))
        for i in range(self.npop):
            m2_pop[i, :t_meals[1]] = m2_params[i, 0]                # before lunch
            m2_pop[i, t_meals[1]:t_meals[2]] = m2_params[i, 1]      # between lunch and dinner
            m2_pop[i, t_meals[2]:] = m2_params[i, 2]                # after dinner

        AC[int(self.tex[0] * 60 / self.ts):int(self.tex[1] * 60 / self.ts)] = self.intex
        D[int(self.textra_CHO*60/self.ts)] = self.Dextra_CHO   

        self.m2_pop = m2_pop
        self.t_meals = t_meals
        self.t_u_basal = t_u_basal
        self.AC = AC
        self.D = D
        
        return True
    
    def load_models(self, fx, hx, nstate, noutput, param_observer_path):
        '''
        :param object fx: state difference equation of observer model, in model.py
        :param object hx: output difference equation
        :param int nstate: number of states in the observer model
        :param int noutput: number of outputs in the observer model
        :param string param_observer_path: path where model parameters are stored
        '''
        UKF_params = pd.read_csv(param_observer_path, sep=';')['value'].to_list()

        self.fx = fx
        self.hx = hx
        self.nstate = nstate
        self.noutput = noutput
        self.UKF_params = UKF_params
        return True
    
    def change_model_params(self, new_UKF_params):
        self.UKF_params = new_UKF_params
        return True
    
    def simulation(self,
                   process_noise = np.diag((np.array([851, 583, 0.1, 1.65e-4, 1.42, 0.93, 1.1, 100, 50, 1.65e-5]))**2), 
                   measure_noise = np.diag([15 ** 2]), **kwargs):
        '''Simulate patients' glucose trajectories and estimate insulin sensitivity (IS) at the same time
        process noise and measurement noise are adjustable but are time invariant
        :param np.ndarray process_noise: initial process noise matrix Q
        :param np.ndarray measurement noise: measurement noise matrix R
        :param **kwargs: additional parameters, including "ini_tauZ" and "save_tauZ" for fx_mod2 and fx_mod3 
        '''
        
        # read patient parameters
        dt = self.ts
        nmeals = len(self.tmeal)
        steps = int(self.dur / self.ts)
        
        # load model parameters
        model_params = pd.read_csv('patient_parameters/params_patientmodel.csv', sep=';')
        k1 = model_params[model_params['param'] == 'k1']['value'].iloc[0]
        k2 = model_params[model_params['param'] == 'k2']['value'].iloc[0]
        k3 = model_params[model_params['param'] == 'k3']['value'].iloc[0]
        k4 = model_params[model_params['param'] == 'k4']['value'].iloc[0]
        Vi = model_params[model_params['param'] == 'Vi']['value'].iloc[0]
        p4 = model_params[model_params['param'] == 'p4']['value'].iloc[0]
        p5 = model_params[model_params['param'] == 'p5']['value'].iloc[0]
        Vg = model_params[model_params['param'] == 'Vg']['value'].iloc[0]
        params = model_params['value'].to_list()
        
        Qb = self.Gb*Vg
        
        points = JulierSigmaPoints(n=self.nstate, kappa=1)

        u_tot = np.zeros((self.npop, nmeals))
        ub_red = np.zeros(self.npop)                     # basal insulin reduction rate

        GI_tot = np.zeros((self.npop, steps))
        BG_tot = np.zeros((self.npop, steps))
        meas_tot = np.zeros((self.npop, steps))
        X_tot = np.zeros((self.npop, steps))
        Y_tot = np.zeros((self.npop, steps))
        Z_tot = np.zeros((self.npop, steps))
        IS_tot = np.zeros((self.npop, steps))

        GI_est_tot = np.zeros((self.npop, steps))
        BG_est_tot = np.zeros((self.npop, steps))
        X_est_tot = np.zeros((self.npop, steps))
        IS_est_tot = np.zeros((self.npop, steps))
        if self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
            tauZ_est_tot = np.zeros((self.npop, steps))

        # insulin input time-series
        u = np.zeros((self.npop, steps))

        for j in range(self.npop):

            print('Patient: ' + str(j+1) + '/' + str(self.npop))

            # extract individual patient parameters
            IS_pat = self.patient_char['IS'].iloc[j]
            Ib_pat = self.patient_char['Ib'].iloc[j]
            ICR_pat = self.patient_char['ICR'].iloc[j]
            CF_pat = self.patient_char['CF'].iloc[j]

            noise_pat = self.noise[:int(self.dur/self.ts), j]

            # extract meal parameters
            m2_pat = self.m2_pop[j, :]

            # compute ub
            ub_pat = (k2 + k3) / k2 * k4 * Vi * self.BW * Ib_pat

            # set up empty list to save insulin doses
            u_pat = []

            # UKF
            kf = UnscentedKalmanFilter(dim_x=self.nstate, dim_z=self.noutput, dt=dt, fx=self.fx, hx=self.hx, points=points)

            # initialize UKF with nominal values
            if self.fx == GIM.fx or self.fx == GIM.fx_mod5:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod1:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod2:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 60, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod3:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 60, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod4:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
                
            kf.Q = process_noise
            kf.R = measure_noise

            outputUKF = np.zeros((steps, self.nstate))
            outputUKF[0, :] = kf.x
            
            # initialize IS basel estimation
            ISb_est = self.IS_nom

            # initialize patient model
            model_pat = np.zeros((steps, 14))
            model_pat[0, :] = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, IS_pat*Ib_pat, Qb, p4/p5*Qb, self.Gb, 0, 0, 0, 0, 0, 0, IS_pat])

            # first BG measurement
            meas_tot[j, 0] = model_pat[0, 6] + noise_pat[0]
            
            for i in range(steps-1):

                # before breakfast, determine basal IS of patient from past 180min
                if i == self.t_meals[0]:
                    tmp = outputUKF[i-int(179/self.ts):i+1, -1]
                    ISb_est = np.mean(tmp)

                # compute meal bolus based on BG and CHO content
                if i in self.t_meals:
                    u[j, i] = IBC.UKF_bolus(outputUKF[i, -1], ISb_est, CF_pat, ICR_pat, self.D[i], meas_tot[j, i], self.Gb)
                    u_pat.append(u[j, i] * 1e-6)
                
                # determine IS and corresponding basal bolus (captured by change in Ib) at bedtime
                if i == self.t_u_basal:
                    IS_bedtime = outputUKF[i, -1]
                    ub_pat = ISb_est / (1/2 * (ISb_est + IS_bedtime)) * ub_pat
                    ub_red[j] = ISb_est / (1/2 * (ISb_est + IS_bedtime))

                # run patient model
                model_pat[i+1, :] = odeint(GIM.ode_system, model_pat[i, :], [0, self.ts], args=(params, m2_pat[i], ub_pat,
                                                                                        u[j, i]/self.ts, self.D[i]/self.ts, self.AC[i], self.BW))[1]

                # create glucose measurement for i+1
                meas_tot[j, i+1] = model_pat[i+1, 6] + noise_pat[i+1]

                # UKF: prediction from step i --> i+1 based on observer model
                
                # UKF update based on measurement i+1
                if self.fx == GIM.fx:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod1:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, AC=self.AC[i], BW=self.BW)
                    kf.update([meas_tot[j, i+1], self.AC[i]])
                elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod5:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW, t = i, ex_end = self.tex[1], IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod3 or self.fx == GIM.fx_mod4:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW, IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                
                # save updated state values
                outputUKF[i+1, :] = kf.x
                
            # save insulin doses
            u_tot[j, :] = np.array(u_pat)

            # save true patient states
            GI_tot[j, :] = model_pat[:, 6]
            BG_tot[j, :] = model_pat[:, 4] / Vg
            X_tot[j, :] = model_pat[:, 3] * (1 + model_pat[:, 10])
            Y_tot[j, :] = model_pat[:, 9]
            Z_tot[j, :] = model_pat[:, 10]
            IS_tot[j, :] = model_pat[:, 13] * (1 + model_pat[:, 10])

            # save estimated states
            GI_est_tot[j, :] = outputUKF[:, 6]
            BG_est_tot[j, :] = outputUKF[:, 4] / Vg
            X_est_tot[j, :] = outputUKF[:, 3]
            
            if self.fx == GIM.fx or self.fx == GIM.fx_mod4 or self.fx == GIM.fx_mod5:
                IS_est_tot[j, :] = outputUKF[:, -1]
            elif self.fx == GIM.fx_mod1:
                IS_est_tot[j, :] = outputUKF[:, -1] * (1 + outputUKF[:, -2])
            elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
                IS_est_tot[j, :] = outputUKF[:, -1]
                tauZ_est_tot[j, :] = outputUKF[:, -2]
        
        if 'save' in kwargs.keys() and 'out' in kwargs.keys():
            out = kwargs['out']
            if kwargs['save'] == 'tauZ':
                np.savetxt(out, tauZ_est_tot, delimiter=';')
            elif kwargs['save'] == 'u':
                np.savetxt(out, u_tot, delimiter=';')
            elif kwargs['save'] == 'ub':
                np.savetxt(out, ub_red, delimiter=';')
            elif kwargs['save'] == 'BG':
                np.savetxt(out, BG_tot, delimiter=';')
            elif kwargs['save'] == 'GI':
                np.savetxt(out, GI_tot, delimiter=';')
            elif kwargs['save'] == 'X':
                np.savetxt(out, X_tot, delimiter=';')
            elif kwargs['save'] == 'BGest':
                np.savetxt(out, BG_est_tot, delimiter=';')
            elif kwargs['save'] == 'GIest':
                np.savetxt(out, GI_est_tot, delimiter=';')
            elif kwargs['save'] == 'Xest':
                np.savetxt(out, X_est_tot, delimiter=';')
            elif kwargs['save'] == 'all':
                out = kwargs['out']
                if not os.path.isdir(out):
                    os.makedirs(out)
                ''' save results '''
                # insulin doses
                np.savetxt(f'{out}/UKF_u.csv', u_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ub.csv', ub_red, delimiter=';')
                # true glucose levels, CGM data and true insulin sensitivity
                np.savetxt(f'{out}/UKF_BG.csv', BG_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GI.csv', GI_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_CGM.csv', meas_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_X.csv', X_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Y.csv', Y_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Z.csv', Z_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_IS.csv', IS_tot, delimiter=';')
                # estimated glucose levels and insulin sensitivity
                np.savetxt(f'{out}/UKF_BGest.csv', BG_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GIest.csv', GI_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Xest.csv', X_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ISest.csv', IS_est_tot, delimiter=';')
        
        return IS_tot, IS_est_tot
    
    def switch_IS_estimation(self, process_noises, measure_noise = np.diag([15 ** 2]), basal_adjust = True, **kwargs):
        '''Simulate patients' glucose trajectories and estimate insulin sensitivity (IS) at the same time
        process noise are time variant
        :param np.ndarray process_noises: a sequence of initial process noise matrix Q, of dimension t * n * n
        :param np.ndarray measurement noise: measurement noise matrix R
        :param bool basal_adjust: If basal insulin rate is adjusted at 22:00.
        :param **kwargs: additional parameters, including "ini_tauZ" and "save_tauZ" for fx_mod2 and fx_mod3 
        '''
        
        # read patient parameters
        dt = self.ts
        nmeals = len(self.tmeal)
        steps = int(self.dur / self.ts)
        
        # load model parameters
        model_params = pd.read_csv('patient_parameters/params_patientmodel.csv', sep=';')
        k1 = model_params[model_params['param'] == 'k1']['value'].iloc[0]
        k2 = model_params[model_params['param'] == 'k2']['value'].iloc[0]
        k3 = model_params[model_params['param'] == 'k3']['value'].iloc[0]
        k4 = model_params[model_params['param'] == 'k4']['value'].iloc[0]
        Vi = model_params[model_params['param'] == 'Vi']['value'].iloc[0]
        p4 = model_params[model_params['param'] == 'p4']['value'].iloc[0]
        p5 = model_params[model_params['param'] == 'p5']['value'].iloc[0]
        Vg = model_params[model_params['param'] == 'Vg']['value'].iloc[0]
        params = model_params['value'].to_list()
        
        Qb = self.Gb*Vg
        
        points = JulierSigmaPoints(n=self.nstate, kappa=1)

        u_tot = np.zeros((self.npop, nmeals))
        ub_red = np.zeros(self.npop)                     # basal insulin reduction rate

        GI_tot = np.zeros((self.npop, steps))
        BG_tot = np.zeros((self.npop, steps))
        meas_tot = np.zeros((self.npop, steps))
        X_tot = np.zeros((self.npop, steps))
        Y_tot = np.zeros((self.npop, steps))
        Z_tot = np.zeros((self.npop, steps))
        IS_tot = np.zeros((self.npop, steps))

        GI_est_tot = np.zeros((self.npop, steps))
        BG_est_tot = np.zeros((self.npop, steps))
        X_est_tot = np.zeros((self.npop, steps))
        IS_est_tot = np.zeros((self.npop, steps))
        if self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
            tauZ_est_tot = np.zeros((self.npop, steps))

        # insulin input time-series
        u = np.zeros((self.npop, steps))

        for j in range(self.npop):

            print('Patient: ' + str(j+1) + '/' + str(self.npop))

            # extract individual patient parameters
            IS_pat = self.patient_char['IS'].iloc[j]
            Ib_pat = self.patient_char['Ib'].iloc[j]
            ICR_pat = self.patient_char['ICR'].iloc[j]
            CF_pat = self.patient_char['CF'].iloc[j]

            noise_pat = self.noise[:int(self.dur/self.ts), j]

            # extract meal parameters
            m2_pat = self.m2_pop[j, :]

            # compute ub
            ub_pat = (k2 + k3) / k2 * k4 * Vi * self.BW * Ib_pat
            
            # set up empty list to save insulin doses
            u_pat = []

            # UKF
            kf = UnscentedKalmanFilter(dim_x=self.nstate, dim_z=self.noutput, dt=dt, fx=self.fx, hx=self.hx, points=points)

            # initialize UKF with nominal values
            if self.fx == GIM.fx or self.fx == GIM.fx_mod5:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod1:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod2:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, ini_tauZ/10, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod3:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, ini_tauZ/10, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod4:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
            
            kf.R = measure_noise

            outputUKF = np.zeros((steps, self.nstate))
            outputUKF[0, :] = kf.x
            
            ISb_est = self.IS_nom

            # initialize patient model
            model_pat = np.zeros((steps, 14))
            model_pat[0, :] = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, IS_pat*Ib_pat, Qb, p4/p5*Qb, self.Gb, 0, 0, 0, 0, 0, 0, IS_pat])

            # first BG measurement
            meas_tot[j, 0] = model_pat[0, 6] + noise_pat[0]

            for i in range(steps-1):
                
                kf.Q = process_noises[i]

                # before breakfast, determine basal IS of patient from past 180min
                if i == self.t_meals[0]:
                    tmp = outputUKF[i-int(179/self.ts):i+1, -1]
                    ISb_est = np.mean(tmp)
                    
                # compute meal bolus based on BG and CHO content
                if i in self.t_meals:
                    u[j, i] = IBC.UKF_bolus(outputUKF[i, -1], ISb_est, CF_pat, ICR_pat, self.D[i], meas_tot[j, i], self.Gb)
                    u_pat.append(u[j, i] * 1e-6)

                # determine IS and corresponding basal bolus (captured by change in Ib) at bedtime
                if i == self.t_u_basal and basal_adjust == True:
                    IS_bedtime = outputUKF[i, -1]
                    ub_pat = ISb_est / (1/2 * (ISb_est + IS_bedtime)) * ub_pat
                    ub_red[j] = ISb_est / (1/2 * (ISb_est + IS_bedtime))

                # run patient model
                model_pat[i+1, :] = odeint(GIM.ode_system, model_pat[i, :], [0, self.ts], args=(params, m2_pat[i], ub_pat,
                                                                                        u[j, i]/self.ts, self.D[i]/self.ts, self.AC[i], self.BW))[1]

                # create glucose measurement for i+1
                meas_tot[j, i+1] = model_pat[i+1, 6] + noise_pat[i+1]

                # UKF: prediction from step i --> i+1 based on observer model
                
                # UKF update based on measurement i+1
                if self.fx == GIM.fx:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod1:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, AC=self.AC[i], BW=self.BW)
                    kf.update([meas_tot[j, i+1], self.AC[i]])
                elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod5:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW, t = i, ex_end = self.tex[1], IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod3 or self.fx == GIM.fx_mod4:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, AC=self.AC[i], BW=self.BW, IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                
                # save updated state values
                outputUKF[i+1, :] = kf.x

            # save insulin doses
            u_tot[j, :] = np.array(u_pat)

            # save true patient states
            GI_tot[j, :] = model_pat[:, 6]
            BG_tot[j, :] = model_pat[:, 4] / Vg
            X_tot[j, :] = model_pat[:, 3] * (1 + model_pat[:, 10])
            Y_tot[j, :] = model_pat[:, 9]
            Z_tot[j, :] = model_pat[:, 10]
            IS_tot[j, :] = model_pat[:, 13] * (1 + model_pat[:, 10])

            # save estimated states
            GI_est_tot[j, :] = outputUKF[:, 6]
            BG_est_tot[j, :] = outputUKF[:, 4] / Vg
            X_est_tot[j, :] = outputUKF[:, 3]
            if self.fx == GIM.fx or self.fx == GIM.fx_mod4 or self.fx == GIM.fx_mod5:
                IS_est_tot[j, :] = outputUKF[:, -1]
            elif self.fx == GIM.fx_mod1:
                IS_est_tot[j, :] = outputUKF[:, -1] * (1 + outputUKF[:, -2])
            elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
                IS_est_tot[j, :] = outputUKF[:, -1]
                tauZ_est_tot[j, :] = outputUKF[:, -2]
                
        if 'save' in kwargs.keys() and 'out' in kwargs.keys():
            out = kwargs['out']
            if kwargs['save'] == 'tauZ':
                np.savetxt(out, tauZ_est_tot, delimiter=';')
            elif kwargs['save'] == 'u':
                np.savetxt(out, u_tot, delimiter=';')
            elif kwargs['save'] == 'ub':
                np.savetxt(out, ub_red, delimiter=';')
            elif kwargs['save'] == 'BG':
                np.savetxt(out, BG_tot, delimiter=';')
            elif kwargs['save'] == 'GI':
                np.savetxt(out, GI_tot, delimiter=';')
            elif kwargs['save'] == 'X':
                np.savetxt(out, X_tot, delimiter=';')
            elif kwargs['save'] == 'BGest':
                np.savetxt(out, BG_est_tot, delimiter=';')
            elif kwargs['save'] == 'GIest':
                np.savetxt(out, GI_est_tot, delimiter=';')
            elif kwargs['save'] == 'Xest':
                np.savetxt(out, X_est_tot, delimiter=';')
            elif kwargs['save'] == 'all':
                out = kwargs['out']
                if not os.path.isdir(out):
                    os.makedirs(out)
                ''' save results '''
                # insulin doses
                np.savetxt(f'{out}/UKF_u.csv', u_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ub.csv', ub_red, delimiter=';')
                # true glucose levels, CGM data and true insulin sensitivity
                np.savetxt(f'{out}/UKF_BG.csv', BG_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GI.csv', GI_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_CGM.csv', meas_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_X.csv', X_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Y.csv', Y_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Z.csv', Z_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_IS.csv', IS_tot, delimiter=';')
                # estimated glucose levels and insulin sensitivity
                np.savetxt(f'{out}/UKF_BGest.csv', BG_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GIest.csv', GI_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Xest.csv', X_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ISest.csv', IS_est_tot, delimiter=';')
        
        return IS_tot, IS_est_tot
    
    def generate_QR(self, initial_process_noise, exercise_process_noise, t_start, t_end):
        '''generate the sequence of process noises Qs and the measurement noise R
        :param np.ndarray initial_process_noise: initial process noise matrix Q
        :param np.ndarray exercise_process_noise: exercise process noise matrix Q
        :param t_ex_start: starting hour of exercise [h]
        :param t_ex_end: ending hour of exercise [h]
        '''
        
        steps = int(self.dur / self.ts)
        if self.fx == GIM.fx or self.fx == GIM.fx_mod5:
            STATE_LIST = np.array(['x1', 'x2', 'I', 'X', 'Q1', 'Q2', 'GI', 'M1', 'M2', 'IS'])
            R = np.diag([15 ** 2])
        elif self.fx == GIM.fx_mod1:
            STATE_LIST = np.array(['x1', 'x2', 'I', 'X', 'Q1', 'Q2', 'GI', 'M1', 'M2', 'PA', 'Y', 'Z', 'IS'])
            R = np.diag((np.array([15, 50])) ** 2)
        elif self.fx == GIM.fx_mod2:
            STATE_LIST = np.array(['x1', 'x2', 'I', 'X', 'Q1', 'Q2', 'GI', 'M1', 'M2', 'tau_Z', 'IS'])
            R = np.diag([15 ** 2])
        elif self.fx == GIM.fx_mod3:
            STATE_LIST = np.array(['x1', 'x2', 'I', 'X', 'Q1', 'Q2', 'GI', 'M1', 'M2', 'Y', 'tau_Z', 'IS'])
            R = np.diag([15 ** 2])
        elif self.fx == GIM.fx_mod4:
            STATE_LIST = np.array(['x1', 'x2', 'I', 'X', 'Q1', 'Q2', 'GI', 'M1', 'M2', 'Y', 'IS'])
            R = np.diag([15 ** 2])
            
        nstate = len(STATE_LIST)
        
        # create a time series of process noises
        Qs = np.zeros((steps, nstate, nstate))
        for i in range(int(t_start*60/self.ts)):
            Qs[i] = initial_process_noise
            
        for i in range(int(t_start*60/self.ts), int(t_end*60/self.ts)):
            Qs[i] = exercise_process_noise
            
        for i in range(int(t_end*60/self.ts), steps):
            Qs[i] = initial_process_noise
        
        return Qs, R
    
    def SBAE(self, initial_process_noise, exercise_process_noise, t_ex_start = 14, t_ex_end = 16, basal_adjust = True, **kwargs):
        '''Switch Back After Exercise Strategy
        :param np.ndarray initial_process_noise: initial process noise matrix Q
        :param np.ndarray exercise_process_noise: exercise process noise matrix Q
        :param int t_ex_start: starting hour of exercise [h]
        :param int t_ex_end: ending hour of exercise [h]
        :param bool basal_adjust: flag variable, if it adjusts the basal insulin infusion
        :param **kwargs: additional parameters, including "ini_tauZ" and "save_tauZ" for fx_mod2 and fx_mod3 
        '''
        process_noises, measurement_noise = self.generate_QR(initial_process_noise, exercise_process_noise, t_ex_start, t_ex_end)
        
        IS_tot, IS_est_tot = self.switch_IS_estimation(process_noises, measurement_noise, basal_adjust, **kwargs)
        
        return IS_tot, IS_est_tot
    
    def SBDE(self, initial_process_noise, exercise_process_noise, t_ex_start = 14, t_early_end = 16, **kwargs):
        '''Switch Back During Exercise Strategy (switch back before end of exercise)
        :param np.ndarray initial_process_noise: initial process noise matrix Q
        :param np.ndarray exercise_process_noise: exercise process noise matrix Q
        :param t_ex_start: starting hour of exercise [h]
        :param t_early_end: early swiching back time [h]
        :param **kwargs: additional parameters, including "ini_tauZ" and "save_tauZ" for fx_mod2 and fx_mod3 
        '''
        process_noises, measurement_noise = self.generate_QR(initial_process_noise, exercise_process_noise, t_ex_start, t_early_end)
        
        IS_tot, IS_est_tot = self.switch_IS_estimation(process_noises, measurement_noise, **kwargs)
        
        return IS_tot, IS_est_tot
    
    def optimize_insulin(self, ini_x, BG_target, sample_time, pred_horizon, D_list, **kwarges):
        '''
        :param np.array ini_x: initial state vector
        :param int: target blood glucose
        :param int: sample time of mpc
        :param int sample_time: sample time of mpc [min]
        :param int pred_horizon: prediction horizon of mpc, number of sample time N
        :param np.array D_list: meal sequence during prediction horizon
        :param **kwargs: additional parameters for observer model, including params, BW, u, t, etc.
        '''
        if 'params' in kwarges.keys():
            Vg = kwarges['params'][4]
        else:
            return
        def model_objective(ub):
            BG_preds = np.zeros(sample_time*pred_horizon)
            x = ini_x
            BG_preds[0] = x[4] / Vg
            
            for i in range(1, sample_time*pred_horizon):
                x = self.fx(x, self.ts, ub = ub, D = D_list[i], **kwarges)
                BG_preds[i] = x[4] / Vg
            
            return np.sum(np.power(np.maximum(BG_preds - BG_target, 0), 2))+ \
                10e4 * np.sum(np.power(np.maximum(BG_target - BG_preds, 0), 2))
        return np.amax([minimize_scalar(model_objective, bounds=(0, 4/60*1e6), method="bounded").x, 0])
    
    def mpc(self, initial_process_noise, exercise_process_noise, period = [20,30], sample_time = 15, pred_horizon = 16, BG_target = 110, 
            t_ex_start = 14, t_ex_end = 16, **kwargs):
        '''
        :param np.ndarray initial_process_noise: initial process noise matrix Q
        :param np.ndarray exercise_process_noise: exercise process noise matrix Q
        :param list[2] period: starting hour and ending hour of model predictive control (mpc) [h]
        :param int sample_time: sample time of mpc [min]
        :param int pred_horizon: prediction horizon of mpc, number of sample time N
        :param int BG_target: target blood glucose [mg/dL]
        :param t_ex_start: starting hour of exercise [h]
        :param t_ex_end: ending hour of exercise [h]
        :param **kwargs: additional parameters, including "ini_tauZ" and "save_tauZ" for fx_mod2 and fx_mod3 
        '''
        # read patient parameters
        dt = self.ts
        nmeals = len(self.tmeal)
        steps = int(self.dur / self.ts)
        
        # load model parameters
        model_params = pd.read_csv('patient_parameters/params_patientmodel.csv', sep=';')
        k1 = model_params[model_params['param'] == 'k1']['value'].iloc[0]
        k2 = model_params[model_params['param'] == 'k2']['value'].iloc[0]
        k3 = model_params[model_params['param'] == 'k3']['value'].iloc[0]
        k4 = model_params[model_params['param'] == 'k4']['value'].iloc[0]
        Vi = model_params[model_params['param'] == 'Vi']['value'].iloc[0]
        p4 = model_params[model_params['param'] == 'p4']['value'].iloc[0]
        p5 = model_params[model_params['param'] == 'p5']['value'].iloc[0]
        Vg = model_params[model_params['param'] == 'Vg']['value'].iloc[0]
        params = model_params['value'].to_list()
        
        Qb = self.Gb*Vg
        
        points = JulierSigmaPoints(n=self.nstate, kappa=1)

        u_tot = np.zeros((self.npop, nmeals))
        ub_red = np.zeros(self.npop)                     # basal insulin reduction rate
        
        mpc_start = int(period[0]*60/self.ts)
        mpc_end = int(period[1]*60/self.ts)
        u_seq = np.zeros((self.npop, mpc_end-mpc_start))

        GI_tot = np.zeros((self.npop, steps))
        BG_tot = np.zeros((self.npop, steps))
        meas_tot = np.zeros((self.npop, steps))
        X_tot = np.zeros((self.npop, steps))
        Y_tot = np.zeros((self.npop, steps))
        Z_tot = np.zeros((self.npop, steps))
        IS_tot = np.zeros((self.npop, steps))

        GI_est_tot = np.zeros((self.npop, steps))
        BG_est_tot = np.zeros((self.npop, steps))
        X_est_tot = np.zeros((self.npop, steps))
        IS_est_tot = np.zeros((self.npop, steps))
        if self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
            tauZ_est_tot = np.zeros((self.npop, steps))

        # insulin input time-series
        u = np.zeros((self.npop, steps))

        for j in range(self.npop):

            print('Patient: ' + str(j+1) + '/' + str(self.npop))

            # extract individual patient parameters
            IS_pat = self.patient_char['IS'].iloc[j]
            Ib_pat = self.patient_char['Ib'].iloc[j]
            ICR_pat = self.patient_char['ICR'].iloc[j]
            CF_pat = self.patient_char['CF'].iloc[j]

            noise_pat = self.noise[:int(self.dur/self.ts), j]

            # extract meal parameters
            m2_pat = self.m2_pop[j, :]

            # compute ub
            ub_pat = (k2 + k3) / k2 * k4 * Vi * self.BW * Ib_pat
            
            # set up empty list to save insulin doses
            u_pat = []

            # UKF
            kf = UnscentedKalmanFilter(dim_x=self.nstate, dim_z=self.noutput, dt=dt, fx=self.fx, hx=self.hx, points=points)

            # initialize UKF with nominal values
            if self.fx == GIM.fx or self.fx == GIM.fx_mod5:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod1:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod2:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, ini_tauZ/10, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod3:
                if 'ini_tauZ' in kwargs.keys():
                    ini_tauZ = kwargs['ini_tauZ']
                else:
                    ini_tauZ = 600
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, ini_tauZ, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, ini_tauZ/10, 1.65e-4]))**2
            elif self.fx == GIM.fx_mod4:
                kf.x = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, self.IS_nom * Ib_pat, Qb, p4 / p5 * Qb, self.Gb, 0, 0, 0, self.IS_nom])
                kf.P *= (np.array([8511, 5833, 1, 1.65e-3, 14.2, 9.3, 11, 1e-3, 1e-3, 1e-3, 1.65e-4]))**2
            
            process_noises, measure_noise = self.generate_QR(initial_process_noise, exercise_process_noise, t_ex_start, t_ex_end)
            kf.R = measure_noise

            outputUKF = np.zeros((steps, self.nstate))
            outputUKF[0, :] = kf.x
            
            ISb_est = self.IS_nom

            # initialize patient model
            model_pat = np.zeros((steps, 14))
            model_pat[0, :] = np.array([ub_pat / k1, ub_pat / (k2 + k3), Ib_pat, IS_pat*Ib_pat, Qb, p4/p5*Qb, self.Gb, 0, 0, 0, 0, 0, 0, IS_pat])

            # first BG measurement
            meas_tot[j, 0] = model_pat[0, 6] + noise_pat[0]

            for i in range(steps-1):
                
                kf.Q = process_noises[i]

                # before breakfast, determine basal IS of patient from past 180min
                if i == self.t_meals[0]:
                    tmp = outputUKF[i-int(179/self.ts):i+1, -1]
                    ISb_est = np.mean(tmp)
                    
                # compute meal bolus based on BG and CHO content
                if i in self.t_meals:
                    u[j, i] = IBC.UKF_bolus(outputUKF[i, -1], ISb_est, CF_pat, ICR_pat, self.D[i], meas_tot[j, i], self.Gb)
                    u_pat.append(u[j, i] * 1e-6)

                # mpc controller in period
                if i >= mpc_start and i < mpc_end:
                    ub_pat = 0
                    if (i - mpc_start) % sample_time == 0:
                        step_ahead = sample_time*pred_horizon
                        if i + step_ahead > steps:
                            D_list = np.zeros(step_ahead)
                            D_list[:steps-i] = self.D[i:steps]/self.ts
                        else:
                            D_list=self.D[i:i+step_ahead]/self.ts
                        ini_x = outputUKF[i, :]
                        if self.fx == GIM.fx:
                            bump_u = self.optimize_insulin(ini_x, BG_target, sample_time, pred_horizon, params=self.UKF_params, u=u[j,i], 
                                                        D_list=D_list, BW=self.BW)
                        elif self.fx == GIM.fx_mod1:
                            bump_u = self.optimize_insulin(ini_x, BG_target, sample_time, pred_horizon, params=self.UKF_params, u=u[j,i], 
                                                        D_list=D_list, AC = self.AC[i], BW=self.BW)
                        elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod5:
                            bump_u = self.optimize_insulin(ini_x, BG_target, sample_time, pred_horizon, params=self.UKF_params, u=u[j,i], 
                                                        D_list=D_list, BW=self.BW, t = i, ex_end = self.tex[1], IS_bas = ISb_est)
                        elif self.fx == GIM.fx_mod3 or self.fx == GIM.fx_mod4:
                            bump_u = self.optimize_insulin(ini_x, BG_target, sample_time, pred_horizon, params=self.UKF_params, u=u[j,i], 
                                                        D_list=D_list, AC = self.AC[i], BW=self.BW, IS_bas = ISb_est)
                        else:
                            bump_u = 0
                    u_seq[j,i-mpc_start] = bump_u
                    ub_pat = bump_u
                
                # run patient model
                model_pat[i+1, :] = odeint(GIM.ode_system, model_pat[i, :], [0, self.ts], args=(params, m2_pat[i], ub_pat,
                                                                                        u[j, i]/self.ts, self.D[i]/self.ts, self.AC[i], self.BW))[1]
                # create glucose measurement for i+1
                meas_tot[j, i+1] = model_pat[i+1, 6] + noise_pat[i+1]

                # UKF: prediction from step i --> i+1 based on observer model
                
                # UKF update based on measurement i+1
                if self.fx == GIM.fx:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod1:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, AC=self.AC[i], BW=self.BW)
                    kf.update([meas_tot[j, i+1], self.AC[i]])
                elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod5:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, BW=self.BW, t = i, ex_end = self.tex[1], IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                elif self.fx == GIM.fx_mod3 or self.fx == GIM.fx_mod4:
                    kf.predict(params=self.UKF_params, ub=ub_pat, u=u[j, i]/self.ts, D=self.D[i]/self.ts, AC=self.AC[i], BW=self.BW, IS_bas = ISb_est)
                    kf.update([meas_tot[j, i+1]])
                
                # save updated state values
                outputUKF[i+1, :] = kf.x

            # save insulin doses
            u_tot[j, :] = np.array(u_pat)

            # save true patient states
            GI_tot[j, :] = model_pat[:, 6]
            BG_tot[j, :] = model_pat[:, 4] / Vg
            X_tot[j, :] = model_pat[:, 3] * (1 + model_pat[:, 10])
            Y_tot[j, :] = model_pat[:, 9]
            Z_tot[j, :] = model_pat[:, 10]
            IS_tot[j, :] = model_pat[:, 13] * (1 + model_pat[:, 10])

            # save estimated states
            GI_est_tot[j, :] = outputUKF[:, 6]
            BG_est_tot[j, :] = outputUKF[:, 4] / Vg
            X_est_tot[j, :] = outputUKF[:, 3]
            if self.fx == GIM.fx or self.fx == GIM.fx_mod4 or self.fx == GIM.fx_mod5:
                IS_est_tot[j, :] = outputUKF[:, -1]
            elif self.fx == GIM.fx_mod1:
                IS_est_tot[j, :] = outputUKF[:, -1] * (1 + outputUKF[:, -2])
            elif self.fx == GIM.fx_mod2 or self.fx == GIM.fx_mod3:
                IS_est_tot[j, :] = outputUKF[:, -1]
                tauZ_est_tot[j, :] = outputUKF[:, -2]
                
        if 'save' in kwargs.keys() and 'out' in kwargs.keys():
            out = kwargs['out']
            if kwargs['save'] == 'tauZ':
                np.savetxt(out, tauZ_est_tot, delimiter=';')
            elif kwargs['save'] == 'u':
                np.savetxt(out, u_tot, delimiter=';')
            elif kwargs['save'] == 'ub':
                np.savetxt(out, ub_red, delimiter=';')
            elif kwargs['save'] == 'BG':
                np.savetxt(out, BG_tot, delimiter=';')
            elif kwargs['save'] == 'GI':
                np.savetxt(out, GI_tot, delimiter=';')
            elif kwargs['save'] == 'X':
                np.savetxt(out, X_tot, delimiter=';')
            elif kwargs['save'] == 'BGest':
                np.savetxt(out, BG_est_tot, delimiter=';')
            elif kwargs['save'] == 'GIest':
                np.savetxt(out, GI_est_tot, delimiter=';')
            elif kwargs['save'] == 'Xest':
                np.savetxt(out, X_est_tot, delimiter=';')
            elif kwargs['save'] == 'all':
                out = kwargs['out']
                if not os.path.isdir(out):
                    os.makedirs(out)
                ''' save results '''
                # insulin doses
                np.savetxt(f'{out}/UKF_u.csv', u_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ub.csv', ub_red, delimiter=';')
                # true glucose levels, CGM data and true insulin sensitivity
                np.savetxt(f'{out}/UKF_BG.csv', BG_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GI.csv', GI_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_CGM.csv', meas_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_X.csv', X_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Y.csv', Y_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Z.csv', Z_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_IS.csv', IS_tot, delimiter=';')
                # estimated glucose levels and insulin sensitivity
                np.savetxt(f'{out}/UKF_BGest.csv', BG_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_GIest.csv', GI_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_Xest.csv', X_est_tot, delimiter=';')
                np.savetxt(f'{out}/UKF_ISest.csv', IS_est_tot, delimiter=';')
        
        return BG_tot, BG_est_tot, u_seq