import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import R2D2
import World
import Plotter


# Instantiate World, Robot, and EKF
R2D2 = R2D2.R2D2()
World = World.World()
Plotter = Plotter.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks)


# Set Timeline
Tf = 20     # Sec
Ts = 0.1    # Sec
time_data = np.arange(0., Tf, Ts)

# Generate Truth
[X, Y, TH] = R2D2.propagateDynamics(time_data)

# fig = plt.figure(1)
# plt.plot(X, t)
# plt.show
# print("Hey.")


# Generate Filter

# Plot
for iter in range(0, X.size):
    Plotter.update(X[iter], Y[iter], TH[iter])



#
# input("Press the any key... goodluck.")