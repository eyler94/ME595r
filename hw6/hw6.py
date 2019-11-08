#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from Ekf_Slam import EKF_SLAM
from R2D2 import R2D2
from World import World
from Plotter import Plotter
from importlib import reload

reload(R2D2)
reload(World)
reload(Plotter)
reload(EKF_SLAM)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


# Instantiate World, Robot, Plotter, and EKF
R2D2 = R2D2.R2D2(FoV=np.pi)
World = World.World(rand=True, num_lm=10)
Plotter = Plotter.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks)
EKF = EKF_SLAM.EkfSlam(R2D2, World)

# Set Timeline
Tf = 50  # Sec
Ts = 0.1  # Sec
time_data = np.arange(0., Tf, Ts)
time_data = time_data.reshape([1, int(Tf / Ts)])

# Generate Truth and Calculate "Noisy Measurements"
X = np.zeros([1, time_data.size])
Y = np.zeros([1, time_data.size])
TH = np.zeros([1, time_data.size])

R = np.zeros([World.Number_Landmarks, time_data.size])  # Range Measurements
PH = np.zeros([World.Number_Landmarks, time_data.size])  # Bearing Measurements

for spot in range(0, int(Tf / Ts)):
    x_y_th = R2D2.propagate_dynamics(time_data[0][spot])
    X[0][spot] = x_y_th[0]
    Y[0][spot] = x_y_th[1]
    TH[0][spot] = x_y_th[2]

    rph = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
    R[:, spot] = rph[0].reshape([World.Number_Landmarks, ])
    PH[:, spot] = rph[1].reshape([World.Number_Landmarks, ])

# Filter Data
MU_X = np.zeros([1, time_data.size])
MU_Y = np.zeros([1, time_data.size])
MU_TH = np.zeros([1, time_data.size])
SIG_X = np.zeros([1, time_data.size])
SIG_Y = np.zeros([1, time_data.size])
SIG_TH = np.zeros([1, time_data.size])

for spot in range(0, int(Tf / Ts)):
    R2D2.update_velocity(time_data[0][spot])
    mu_sig = EKF.update(EKF.mu, EKF.SIG, R2D2.v_c, R2D2.omega_c, R[:, spot], PH[:, spot])
    MU_X[0][spot] = mu_sig[0][0]
    MU_Y[0][spot] = mu_sig[0][1]
    MU_TH[0][spot] = mu_sig[0][2]
    SIG_X[0][spot] = 2 * np.sqrt(mu_sig[1][0][0])
    SIG_Y[0][spot] = 2 * np.sqrt(mu_sig[1][1][1])
    SIG_TH[0][spot] = 2 * np.sqrt(mu_sig[1][2][2])

# Plot
plt.ion()
plt.interactive(False)

# for spot in range(0, X.size):
#     Plotter.update_with_path(X[0][spot], Y[0][spot], TH[0][spot], X[0][:spot], Y[0][:spot], MU_X[0][:spot], MU_Y[0][:spot])

fig3 = plt.figure(3)
fig3.clf()
plt.plot(X.T, Y.T)
plt.plot(MU_X.T, MU_Y.T)
plt.title('Path')

fig4 = plt.figure(4)
fig4.clf()
plt.plot(time_data[0], MU_X[0] - X[0])
plt.plot(time_data[0], SIG_X[0])
plt.plot(time_data[0], -SIG_X[0])
plt.title('Error in X')

fig5 = plt.figure(5)
fig5.clf()
plt.plot(time_data[0], MU_Y[0] - Y[0])
plt.plot(time_data[0], SIG_Y[0])
plt.plot(time_data[0], -SIG_Y[0])
plt.title('Error in Y')

fig6 = plt.figure(6)
fig6.clf()
plt.plot(time_data[0], wrapper(MU_TH[0] - TH[0]))
plt.plot(time_data[0], SIG_TH[0])
plt.plot(time_data[0], -SIG_TH[0])
plt.title('Error in Theta')
plt.show()