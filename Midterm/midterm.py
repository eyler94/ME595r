#!/usr/bin/env python3

import numpy as np
import scipy.io as spio
from importlib import reload
import matplotlib.pyplot as plt
from time import sleep
import sys

sys.path.append('..')
from Midterm import Quad  # import quadrotor
from Midterm import QWorld  # import World
from Plotter import Plotter
from EKF import EKF

reload(Quad)
reload(QWorld)
reload(Plotter)
reload(EKF)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


# Instantiate Quadcopter, World, and Plotter
quad = Quad.quadrotor()
World = QWorld.World()
Pltr = Plotter.Plotter(width=30, height=30, lm=World.Landmarks)

# Reading in data from the mat file provided
mat = spio.loadmat('midterm_data.mat')
state = mat['X_tr']  # x, y, theta
X = state[0]
Y = state[1]
TH = state[2]
omega_commanded = mat['om_c']
omega_actual = mat['om']
velocity_commanded = mat['v_c']
velocity_actual = mat['v']
range_tr = mat['range_tr']
bearing_tr = mat['bearing_tr']
time = mat['t']

# Filter Data
MU_X = np.zeros([1, time.shape[1]])
MU_Y = np.zeros([1, time.shape[1]])
MU_TH = np.zeros([1, time.shape[1]])
SIG_X = np.zeros([1, time.shape[1]])
SIG_Y = np.zeros([1, time.shape[1]])
SIG_TH = np.zeros([1, time.shape[1]])

# Plot Data
plt.ion()
plt.interactive(False)

for iter in range(0, int(quad.tf / quad.ts)):
    mu_sig = EKF.update(EKF.mu, EKF.SIG, velocity_commanded[iter], omega_commanded[iter], range_tr[:, iter],
                        bearing_tr[:, iter])
    MU_X[0][iter] = mu_sig[0][0]
    MU_Y[0][iter] = mu_sig[0][1]
    MU_TH[0][iter] = mu_sig[0][2]
    SIG_X[0][iter] = 2 * np.sqrt(mu_sig[1][0][0])
    SIG_Y[0][iter] = 2 * np.sqrt(mu_sig[1][1][1])
    SIG_TH[0][iter] = 2 * np.sqrt(mu_sig[1][2][2])

for iter in range(0, state.shape[1]):
    print("Iter", iter)
    Pltr.update(state[0][iter], state[1][iter], state[2][iter], 1)

fig3 = plt.figure(3)
fig3.clf()
plt.plot(X.T, Y.T)
# plt.plot(MU_X.T, MU_Y.T)
plt.title('Path')
#
fig4 = plt.figure(4)
fig4.clf()
# plt.plot(time_data[0], MU_X[0] - X[0])
# plt.plot(time_data[0], SIG_X[0])
# plt.plot(time_data[0], -SIG_X[0])
plt.title('Error in X')

fig5 = plt.figure(5)
fig5.clf()
# plt.plot(time_data[0], MU_Y[0] - Y[0])
# plt.plot(time_data[0], SIG_Y[0])
# plt.plot(time_data[0], -SIG_Y[0])
plt.title('Error in Y')

fig6 = plt.figure(6)
fig6.clf()
# plt.plot(time_data[0], wrapper(MU_TH[0] - TH[0]))
# plt.plot(time_data[0], SIG_TH[0])
# plt.plot(time_data[0], -SIG_TH[0])
plt.title('Error in Theta')
plt.show()

print("bah")
