import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')
from Kalman.kalman import Kf
from UUV import UUV

# Instantiate the classes
UUV = UUV()
Kf = Kf(UUV.mu, UUV.SIG, UUV.F_norm, UUV.z, UUV.A, UUV.B, UUV.C, UUV.Q, UUV.R)

first = True

time_data = np.arange(0., UUV.tf, UUV.ts)
for time in time_data:

    # Determine Force
    if time < UUV.t1:
        u = UUV.F_norm
    elif time < UUV.t2:
        u = 0.
    elif time < UUV.t3:
        u = -UUV.F_norm
    else:
        u = 0.

    # Propagate Dynamics
    UUV.propagate_dynamics(u)

    # Collect Measurements
    UUV.z = UUV.collect_measurements()

    # Update Filter
    [UUV.mu, UUV.SIG] = Kf.update(Kf.mu, UUV.SIG, u, UUV.z)

    # Save Data
    if first:
        x_data = UUV.X
        mu_data = UUV.mu
        z_data = UUV.z
        k_data = Kf.K
        SIG = np.diag(UUV.SIG).reshape([2, 1])
        Standard_deviation = np.array([[2 * np.sqrt(SIG[0][0])],
                                       [2 * np.sqrt(SIG[1][0])]])
        SIG_data = Standard_deviation
        first = False
    else:
        x_data = np.hstack([x_data, UUV.X])
        mu_data = np.hstack([mu_data, UUV.mu])
        z_data = np.hstack([z_data, UUV.z])
        k_data = np.hstack([k_data, Kf.K])
        SIG = np.diag(UUV.SIG).reshape([2, 1])
        Standard_deviation = np.array([[2 * np.sqrt(SIG[0][0])],
                                       [2 * np.sqrt(SIG[1][0])]])
        SIG_data = np.hstack([SIG_data, Standard_deviation])

plot = True
if plot:
        # Plot Truth, Estimates, and Measurements
    fig = plt.figure(1)
    time_data = time_data.reshape([1000, 1])
    plt.plot(time_data, x_data[0, :].T)
    plt.plot(time_data, mu_data[0, :].T)
    plt.plot(time_data, z_data[0, :].T)
    plt.plot(time_data, x_data[1, :].T)
    plt.plot(time_data, mu_data[1, :].T)
    plt.legend(['position truth', 'position estimate', 'measurements', 'velocity truth', 'velocity estimate'])

    # Plot Estimation error	and	error covariance versus	time
    fig = plt.figure(2)
    plt.plot(time_data, mu_data[0, :].T - x_data[0, :].T)
    plt.plot(time_data, mu_data[1, :].T - x_data[1, :].T)
    plt.plot(time_data, SIG_data[0, :].T)
    plt.plot(time_data, SIG_data[1, :].T)
    plt.plot(time_data, -SIG_data[0, :].T)
    plt.plot(time_data, -SIG_data[1, :].T)
    plt.legend(['Estimate Error for position', 'Estimate error for velocity', 'Error covariance for position',
                'Error covariance for velocity', 'Lower error covariance for position',
                'Lower error covariance for velocity'])

    # Plot Kalman Gain vs. Time
    fig = plt.figure(3)
    plt.plot(time_data, k_data[0, :].T)
    plt.plot(time_data, k_data[1, :].T)
    plt.legend(['Kalman Gain for postion', 'Kalman Gain for velocity'])
    plt.show()

#
