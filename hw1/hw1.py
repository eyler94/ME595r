import sys

sys.path.append('..')
from UUV import UUV
from Kalman.kalman import Kf
import matplotlib.pyplot as plt
import numpy as np

UUV = UUV()

z = 0.

Kf = Kf(UUV.mu, UUV.SIG, UUV.F_norm, z, UUV.A, UUV.B, UUV.C, UUV.Q, UUV.R)
X_1 = np.array([[0.],
                [0.]])

for time in range(0, int(UUV.tf * 100), int(UUV.ts * 100)):
    # Propagate Dynamics
    if time < UUV.t1 * 100:
        u = UUV.F_norm
    elif time < UUV.t2 * 100:
        u = 0.
    elif time < UUV.t3 * 100:
        u = UUV.F_norm
    else:
        u = 0.
    mu = UUV.propagate_dynamics(u)
    # Collect Measurements
    UUV.collect_measurements()
    # Update Filter
    # Save Data

# Plot Data
# plt.plot(truth[:iter, 0], truth[:iter, 1])
# plt.axis([-1, np.amax(z_log) + 1, -1, np.amax(z_log) / 2.])
# plt.plot(mu_log[:iter, 0], mu_log[:iter, 1])
# plt.plot(z_log[:iter, 0], z_log[:iter, 1])
# plt.show()
