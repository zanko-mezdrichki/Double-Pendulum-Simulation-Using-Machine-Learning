import numpy as np

def obtain_config():
    parameters = {}
    print("Insert all the values, if the input is not valid, the default parameters will be used")
    try:
        parameters['L1'] = float(input("Length of the first pendulum:"))
        parameters['L2'] = float(input("Length of the second pendulum:"))
        parameters['m1'] = float(input("Mass of the first pendulum:"))
        parameters['m2'] = float(input("Mass of the second pendulum:"))
        parameters['g'] = float(input("Gravitational acceleration coefficient:"))
    except Exception:
        print("Using default physical parameters.")
        parameters = {'L1': 1., 'L2': 1., 'm1': 1., 'm2': 1., 'g': 9.81}

    print("Enter the initial conditions of the angles")
    try:
        parameters['theta1_0'] = float(input("First angle (rad):"))
        parameters['theta2_0'] = float(input("Second angle (rad):"))
        parameters['dtheta1_0'] = float(input("First angular velocity (rad/s):"))
        parameters['dtheta2_0'] = float(input("Second angular velocity (rad/s):"))
    except Exception:
        print("Using default angles and angular velocities.")
        parameters['theta1_0'] = np.pi/2
        parameters['theta2_0'] = np.pi/2
        parameters['dtheta1_0'] = 0 
        parameters['dtheta2_0'] = 0 

    print("Enter the PINN settings")
    try:
        parameters['t_max'] = int(input("Duration of the simulation:"))
        parameters['n_points'] = int(input("Number of time points used:"))
        parameters['epochs'] = int(input("Epochs:"))
        parameters['lr'] = float(input("Learning Rate:"))

    except Exception:
        print("Using default PINN parameters.")  
        parameters['t_max'] = 10.0
        parameters['n_points'] = 1001 
        parameters['epochs'] = 10000
        parameters['lr'] = 1e-3
    return parameters