import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import R2D2
import World
import Plotter


# Instantiate World, Robot, Plotter, and EKF
R2D2 = R2D2.R2D2()
World = World.World()
Plotter = Plotter.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks)
EKF = EKF()

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


# Generate Filter


# Plot
for iter in range(0, X.size):
    Plotter.update(X[0][iter], Y[0][iter], TH[0][iter])


