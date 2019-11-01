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
from Midterm import EIFilter  # import EIF
from Plotter import Plotter

reload(Quad)
reload(QWorld)
reload(Plotter)
reload(EIFilter)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


# Instantiate Quadcopter, World, and Plotter
quad = Quad.quadrotor()
World = QWorld.World()
Pltr = Plotter.Plotter(width=30, height=30, lm=World.Landmarks)
eif = EIFilter.EIF(quad, World)

# Reading in data from the mat file provided
mat = spio.loadmat('midterm_data.mat')
state = mat['X_tr']  # x, y, theta
X = state[0]
Y = state[1]
TH = state[2]
omega_com = mat['om_c']
omega_act = mat['om']
velocity_com = mat['v_c']
velocity_act = mat['v']
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
xi_vec = np.zeros([3, time.shape[1]])

for iter in range(0, state.shape[1]):
    eif.update(eif.xi, eif.Omega, velocity_com[0, iter], omega_com[0, iter], range_tr[iter, :], bearing_tr[iter, :])
    MU_X[0][iter] = eif.mu[0]
    MU_Y[0][iter] = eif.mu[1]
    MU_TH[0][iter] = eif.mu[2]
    SIG_X[0][iter] = 2 * np.sqrt(eif.SIG[0][0])
    SIG_Y[0][iter] = 2 * np.sqrt(eif.SIG[1][1])
    SIG_TH[0][iter] = 2 * np.sqrt(eif.SIG[2][2])
    xi_vec[:, iter] = eif.xi.T

# Plot Data
plt.ion()
plt.interactive(False)

for iter in range(0, state.shape[1]):
    Pltr.update_with_path(state[0][iter], state[1][iter], state[2][iter], X[:iter], Y[:iter], MU_X[0][:iter],
                          MU_Y[0][:iter])

fig2 = plt.figure(2)
fig2.clf()
plt.plot(time[0], xi_vec[0])
plt.plot(time[0], xi_vec[1])
plt.plot(time[0], xi_vec[2])
plt.legend(['xi1', 'xi2', 'xi3'])
plt.title('Xi vector')

fig3 = plt.figure(3)
fig3.clf()
plt.plot(X.T, Y.T)
plt.plot(MU_X.T, MU_Y.T)
plt.title('Path')

fig4 = plt.figure(4)
fig4.clf()
plt.plot(time[0], MU_X[0] - X)
plt.plot(time[0], SIG_X[0])
plt.plot(time[0], -SIG_X[0])
plt.title('Error in X')

fig5 = plt.figure(5)
fig5.clf()
plt.plot(time[0], MU_Y[0] - Y)
plt.plot(time[0], SIG_Y[0])
plt.plot(time[0], -SIG_Y[0])
plt.title('Error in Y')

fig6 = plt.figure(6)
fig6.clf()
plt.plot(time[0], wrapper(MU_TH[0] - TH))
plt.plot(time[0], SIG_TH[0])
plt.plot(time[0], -SIG_TH[0])
plt.title('Error in Theta')
plt.show()
