import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import R2D2
import World
import Plotter
from EKF import EKF
from importlib import reload
reload(R2D2)
reload(World)
reload(Plotter)
reload(EKF)

# Instantiate World, Robot, Plotter, and EKF
R2D2 = R2D2.R2D2()
World = World.World()
Plotter = Plotter.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks)
EKF = EKF.EKF(R2D2, World)

# Set Timeline
Tf = 20     # Sec
Ts = 0.1    # Sec
time_data = np.arange(0., Tf, Ts)
time_data = time_data.reshape([1, 200])

# Generate Truth and Calculate "Noisy Measurements"
X = np.zeros([1, time_data.size])
Y = np.zeros([1, time_data.size])
TH = np.zeros([1, time_data.size])

R = np.zeros([World.Number_Landmarks, time_data.size]) # Range Measurements
PH = np.zeros([World.Number_Landmarks, time_data.size]) # Bearing Measurements

X[0][0] = R2D2.x0
Y[0][0] = R2D2.y0
TH[0][0] = R2D2.theta0
rph = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
R[:, 0] = rph[0].reshape([3, ])
PH[:, 0] = rph[1].reshape([3, ])

for iter in range(1, 200):
    xyth = R2D2.propagate_dynamics(time_data[0][iter])
    X[0][iter] = xyth[0]
    Y[0][iter] = xyth[1]
    TH[0][iter] = xyth[2]

    rph = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
    R[:, iter] = rph[0].reshape([3, ])
    PH[:, iter] = rph[1].reshape([3, ])


# Filter Data
MU_X = np.zeros([1, time_data.size])
MU_Y = np.zeros([1, time_data.size])
MU_TH = np.zeros([1, time_data.size])


R2D2.update_velocity(time_data[0][0])
mu_sig = EKF.update(EKF.mu, EKF.SIG, R2D2.v_c, R2D2.omega_c, R[:, 0], PH[:, 0])
MU_X[0][0] = mu_sig[0][0]
MU_Y[0][0] = mu_sig[0][1]
MU_TH[0][0] = mu_sig[0][2]


for iter in range(1, 200):
    R2D2.update_velocity(time_data[0][iter])
    mu_sig = EKF.update(EKF.mu, EKF.SIG, R2D2.v_c, R2D2.omega_c, R[:, iter], PH[:, iter])
    MU_X[0][iter] = mu_sig[0][0]
    MU_Y[0][iter] = mu_sig[0][1]
    MU_TH[0][iter] = mu_sig[0][2]

# Plot
plt.ion()
for iter in range(0, X.size):
    Plotter.update(X[0][iter], Y[0][iter], TH[0][iter])


fig3 = plt.figure(3)
fig3.clf()
plt.plot(X.T, Y.T)
plt.plot(MU_X.T, MU_Y.T)
plt.title('Path')

fig4 = plt.figure(4)
fig4.clf()
plt.title('X')


