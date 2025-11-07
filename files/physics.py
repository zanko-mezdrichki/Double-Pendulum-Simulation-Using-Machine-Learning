import numpy as np
import sympy as smp
from scipy.integrate import odeint

# Symbols
t, g = smp.symbols('t g')
m1, m2 = smp.symbols('m1 m2')
L1, L2 = smp.symbols('L1 L2')
theta1, theta2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)

theta1 = theta1(t)
theta2 = theta2(t)

dtheta1 = smp.diff(theta1, t)
dtheta2 = smp.diff(theta2, t)
ddtheta1 = smp.diff(dtheta1, t)
ddtheta2 = smp.diff(dtheta2, t)

x1 = L1*smp.sin(theta1)
y1 = -L1*smp.cos(theta1)
x2 = L1*smp.sin(theta1) + L2*smp.sin(theta2)
y2 = -L1*smp.cos(theta1) - L2*smp.cos(theta2)

# Defining the Lagrangian
T1 = 0.5*m1*(smp.diff(x1, t)**2 + smp.diff(y1, t)**2)
T2 = 0.5*m2*(smp.diff(x2, t)**2 + smp.diff(y2, t)**2)
T = T1 + T2
V1 = m1*g*y1
V2 = m2*g*y2
V = V1 + V2
L = T - V

# Euler-Lagrange equation
EL_1 = (smp.diff(smp.diff(L, dtheta1), t) - smp.diff(L, theta1)).simplify() #simplify= writing it in the simplest way
EL_2 = (smp.diff(smp.diff(L, dtheta2), t) - smp.diff(L, theta2)).simplify()
eq_1 = smp.Eq(EL_1, 0)
eq_2 = smp.Eq(EL_2, 0)
solutions = smp.solve([eq_1, eq_2], [ddtheta1, ddtheta2], simplify=False, rational=False)

# From two second order ODEs to 4 first order ODEs and making them usable from numpy
dz1dt_f = smp.lambdify((t, g, m1, m2, L1, L2, theta1, theta2, dtheta1, dtheta2), solutions[ddtheta1])
dz2dt_f = smp.lambdify((t, g, m1, m2, L1, L2, theta1, theta2, dtheta1, dtheta2), solutions[ddtheta2]) #ddtheta but numerical
dtheta1dt_f = smp.lambdify(dtheta1, dtheta1) #dtheta but numerical
dtheta2dt_f = smp.lambdify(dtheta2, dtheta2)

def dSdt(S, t, g, m1, m2, L1, L2):
    #function(s) in a list  to be solved(4 ODEs)
    theta1, z1, theta2, z2 = S
    return [
        dtheta1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, theta1, theta2, z1, z2),
        dtheta2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, theta1, theta2, z1, z2),
    ]

def true_simulation(config):
    g_val = config['g']
    m1_val = config['m1']
    m2_val = config['m2']
    L1_val = config['L1']
    L2_val = config['L2']
    
    y0 = [
        config['theta1_0'],
        config['dtheta1_0'],
        config['theta2_0'],
        config['dtheta2_0']
    ]

    t_val = np.linspace(0, config['t_max'], config['n_points']) #list of times for which to find a solution(step=(t_max-0)/n_points)

    
    args_tuple = (g_val, m1_val, m2_val, L1_val, L2_val)

    ans = odeint(dSdt, y0=y0, t=t_val, args=args_tuple) #(func, initial cond, time points to solve the ODE(s), arg(in tuple))

    return t_val, ans