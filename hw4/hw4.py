import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')
from R2D2 import R2D2
from World import World
from Plotter import PlotterParticles as PLTTER
from MCL import MCL
from importlib import reload
from time import sleep

reload(R2D2)
reload(World)
reload(PLTTER)
reload(MCL)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


# Instantiate World, Robot, Plotter, and MCL
R2D2 = R2D2.R2D2()
World = World.World()
MCL = MCL.MCL(R2D2, World)
Plotter = PLTTER.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks, MCL.particles)

# Set Timeline
Tf = 20  # Sec
Ts = 0.1  # Sec
time_data = np.arange(0., Tf, Ts)
time_data = time_data.reshape([1, int(Tf / Ts)])

# Generate Truth and Calculate "Noisy Measurements"
X = np.zeros([1, time_data.size])
Y = np.zeros([1, time_data.size])
TH = np.zeros([1, time_data.size])

R = np.zeros([World.Number_Landmarks, time_data.size])  # Range Measurements
PH = np.zeros([World.Number_Landmarks, time_data.size])  # Bearing Measurements

for iter in range(0, int(Tf / Ts)):
    xyth = R2D2.propagate_dynamics(time_data[0][iter])
    X[0][iter] = xyth[0]
    Y[0][iter] = xyth[1]
    TH[0][iter] = xyth[2]

    rph = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
    R[:, iter] = rph[0].reshape([World.Number_Landmarks, ])
    PH[:, iter] = rph[1].reshape([World.Number_Landmarks, ])

# Filter Data
Particles = np.zeros([time_data.size, 4, MCL.num_particles])
MU_X = np.zeros([1, time_data.size])
MU_Y = np.zeros([1, time_data.size])
MU_TH = np.zeros([1, time_data.size])
SIG_X = np.zeros([1, time_data.size])
SIG_Y = np.zeros([1, time_data.size])
SIG_TH = np.zeros([1, time_data.size])

for iter in range(0, int(Tf / Ts)):
    # print("time", time_data[0][iter])
    R2D2.update_velocity(time_data[0][iter])
    MCL.update(R2D2.v_c, R2D2.omega_c, R[:, iter], PH[:, iter])
    Particles[iter] = MCL.particles
    MU_X[0][iter] = MCL.mu_x
    MU_Y[0][iter] = MCL.mu_y
    MU_TH[0][iter] = MCL.mu_th
    SIG_X[0][iter] = 2 * MCL.sig_x
    SIG_Y[0][iter] = 2 * MCL.sig_y
    SIG_TH[0][iter] = 2 * MCL.sig_th

# Plot
plt.ion()
plt.interactive(False)

sleep(10)
for iter in range(0, X.size):
    Plotter.update(X[0][iter], Y[0][iter], TH[0][iter], Particles[iter])
    sleep(0.1)

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
#
fig5 = plt.figure(5)
fig5.clf()
plt.plot(time_data[0], MU_Y[0] - Y[0])
plt.plot(time_data[0], SIG_Y[0])
plt.plot(time_data[0], -SIG_Y[0])
plt.title('Error in Y')
#
fig6 = plt.figure(6)
fig6.clf()
plt.plot(time_data[0], wrapper(MU_TH[0] - TH[0]))
plt.plot(time_data[0], SIG_TH[0])
plt.plot(time_data[0], -SIG_TH[0])
plt.title('Error in Theta')
plt.show()
