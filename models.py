import numpy as np


''' observer model '''
def fx(x, dt, params, ub, u, D, BW):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, IS = x

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  IS], dtype=float)

    return F

'''modified observer model 1. Add exercise-directed IS increase'''
def fx_mod1(x, dt, params, ub, u, D, AC, BW):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, tau_AC, tau_Z, b, aY, n1, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, PA, Y, Z, IS = x
    fY = (Y / aY) ** n1 / (1 + (Y / aY) ** n1)

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + (1 + Z) * X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  AC,
                  Y + dt * (1 / tau_AC * (AC - Y)),
                  Z + dt * (b * fY * Y - 1 / tau_Z * (1 - fY) * Z),
                  IS], dtype=float)

    return F

'''modified observer model 2. (exercise ending) Time percieved forcing drop method'''
def fx_mod2(x, dt, t, params, ub, u, D, BW, ex_end, IS_bas):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, tau_Z, IS = x

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  tau_Z,
                  IS + dt * (- 1 / tau_Z) * (t >= int(ex_end*60/dt)) * (IS - IS_bas)], dtype=float)

    return F

'''modified observer model 3. Accelerometer perceived forcing drop'''
def fx_mod3(x, dt, params, ub, u, D, AC, BW, IS_bas):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, tau_AC, aY, n1, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, Y, tau_Z, IS = x
    fY = (Y / aY) ** n1 / (1 + (Y / aY) ** n1)

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  Y + dt * (1 / tau_AC * (AC - Y)),
                  tau_Z,
                  IS + dt * (- 1 / tau_Z) * (1-fY) * (IS - IS_bas)], dtype=float)

    return F

'''modified observer model 4. accelerometer perceived forcing drop with fixed tauZ value'''
def fx_mod4(x, dt, params, ub, u, D, AC, BW, IS_bas):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, tau_AC, tau_Z, aY, n1, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, Y, IS = x
    fY = (Y / aY) ** n1 / (1 + (Y / aY) ** n1)

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  Y + dt * (1 / tau_AC * (AC - Y)),
                  IS + dt * (- 1 / tau_Z) * (1-fY) * (IS - IS_bas)], dtype=float)

    return F

'''modified observer model 5. (exercise ending) Time percieved forcing drop method with fixed tauZ value'''
def fx_mod5(x, dt, t, params, ub, u, D, BW, ex_end, IS_bas):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, m2, f, tau_Z, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, IS = x

    F = np.array([x1 + dt * (u + ub - k1 * x1),
                  x2 + dt * (k1 * x1 - (k2 + k3) * x2),
                  I + dt * (k2 / (Vi * BW) * x2 + - k4 * I),
                  X + dt * (IS * p2 * I - p2 * X),
                  Q1 + dt * (EGP0 + f * m2 / BW * M2 + p5 * Q2 - (p1 + X + p4) * Q1),
                  Q2 + dt * (p4 * Q1 - p5 * Q2),
                  GI + dt * 1 / tauG * (Q1/Vg - GI),
                  M1 + dt * (D - m1 * M1),
                  M2 + dt * (m1 * M1 - m2 * M2),
                  IS + dt * (- 1 / tau_Z) * (t >= int(ex_end*60/dt)) * (IS - IS_bas)], dtype=float)

    return F

def hx(x):
    return np.array([x[6]])

def hx_mod1(x):
    return np.array([x[6], x[9]])

def hx_mod2(x):
    return np.array([x[6]])

def hx_mod3(x):
    return np.array([x[6]])

def hx_mod4(x):
    return np.array([x[6]])

def hx_mod5(x):
    return np.array([x[6]])

''' patient model '''

def ode_system(x, t, params, m2, ub, u, D, AC, BW):
    p1, p2, p4, p5, Vg, k1, k2, k3, k4, Vi, m1, f, tau_AC, tau_Z, b, q1, q2, q3, q4, aY, n1, EGP0, tauG = params

    x1, x2, I, X, Q1, Q2, GI, M1, M2, Y, Z, GU, GP, IS = x
    fY = (Y / aY) ** n1 / (1 + (Y / aY) ** n1)

    dxdt = [-k1*x1 + u + ub,
            k1*x1 - (k2+k3)*x2,
            k2/(Vi*BW)*x2 - k4*I,
            IS*p2*I - p2*X,
            -(p1+GU-GP+(1+Z)*X+p4)*Q1 + p5*Q2 + EGP0 + f*m2/BW * M2,
            p4*Q1 - p5*Q2,
            1/tauG * (Q1/Vg - GI),
            -m1*M1 + D,
            m1*M1 - m2*M2,
            1/tau_AC * (AC - Y),
            b * fY * Y - 1/tau_Z * (1-fY) * Z,
            q1 * fY * Y - q2 * GU,
            q3 * fY * Y - q4 * GP,
            0]

    return dxdt


''' basal insulin '''

def Ib_model(Ib_current, t, Ib_target):
    n = 0.04
    return n * (Ib_target - Ib_current)