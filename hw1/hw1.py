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
    UUV.mu = UUV.propagate_dynamics(u)

    # Collect Measurements
    UUV.z = UUV.collect_measurements()

    # Update Filter
    [UUV.mu, UUV.SIG] = Kf.update(UUV.mu, UUV.SIG, u, UUV.z)

    # Save Data
    if first:
        x_data = UUV.X
        mu_data = UUV.mu
        z_data = UUV.z
        k_data = Kf.K
        first = False
    else:
        x_data = np.hstack([x_data, UUV.X])
        mu_data = np.hstack([mu_data, UUV.mu])
        z_data = np.hstack([z_data, UUV.z])
        k_data = np.hstack([k_data, Kf.K])

# Plot Data
fig = plt.figure(1)
time_data = time_data.reshape([1000, 1])
plt.plot(time_data, x_data[0, :].T)
plt.plot(time_data, mu_data[0, :].T)
plt.plot(time_data, z_data[0, :].T)
plt.plot(time_data, x_data[1, :].T)
plt.plot(time_data, mu_data[1, :].T)
plt.legend(['position truth', 'position estimate', 'measurements', 'velocity truth', 'velocity estimate'])
plt.show()

fig = plt.figure(2)
plt.plot(time_data, k_data[0, :].T)
plt.plot(time_data, k_data[1, :].T)
plt.legend(['Kalman Gain for postion','Kalman Gain for velocity'])
plt.show()
