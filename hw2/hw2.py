import numpy as np
import sys
sys.path.append('..')
import R2D2
import World
import Plotter as Plotter


# Instantiate World, Robot, and EKF
R2D2 = R2D2.R2D2()
World = World.World()
plot = Plotter.Plotter(R2D2.x0, R2D2.y0, R2D2.theta0, World.width, World.height, World.Landmarks)


# Set Timeline
Tf = 20     # Sec
Ts = 0.1    # Sec
time_data = np.arange(0., Tf, Ts)

# Generate Truth

# Generate Filter

# Plot
import matplotlib.pyplot as plt
import time
circle = np.arange(0., 2*np.pi, 0.01)
x = np.cos(circle)
x = np.hstack([0., x])
y = np.sin(circle)
y = np.hstack([0., y])

plt.ion()

for iter in np.arange(0., 50., 0.75):
    Xr = np.cos(2*np.pi*iter/50.)
    Yr = np.sin(2*np.pi*iter/50.)
    Theta = np.pi/2.+2*np.pi*iter/50.
    hmt = np.array([[np.cos(Theta), -np.sin(Theta), Xr], [np.sin(Theta), np.cos(Theta), Yr], [0., 0., 1.]])
    points = hmt @ np.array([x, y, np.ones([1, x.size])])
    plt.plot(points[0].T, points[1].T)
    plt.axis([-World.width/2., World.width/2., -World.height/2., World.height/2.])
    plt.draw()
    plt.pause(0.00001)
    plt.clf()

input("Press the any key... goodluck.")

