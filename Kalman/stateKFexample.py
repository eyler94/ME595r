import numpy as np
import matplotlib.pyplot as plt
from kalman import Kf

# This is going to model a projectile

size = 250

k = 0
b = 0
g = -9.81
u = np.array([[0., g]]).T
m = 1
V0 = 50
Ts = 0.04
theta = np.radians(15)

X_1 = np.array([[0, np.cos(theta) * V0, 0, np.sin(theta) * V0]]).T
X = np.array([[0., 0., 0., 0.]]).T

a = np.array([[1., Ts], [0, (1. - b)]])
A = np.zeros([4, 4])
A[0:2, 0:2] = a
A[2:, 2:] = a

b = np.array([[(Ts ** 2.) / 2], [Ts]])
B = np.array([[(Ts ** 2.) / 2, 0.], [Ts, 0.], [0., (Ts ** 2.) / 2], [0., Ts]])

print(B)

c = np.array([1., 0.])
C = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.]])

truth = np.zeros([size, 2])
truth[0, :] = X_1.item(0), X_1.item(2)

# Model
SIG = np.eye(4) * 1e-2  # model uncertainty
mu = X_1  # Original estimate
Q = np.eye(4) * 1e-2  # State transition uncertainty
mu_log = np.zeros([size, 2])
mu_log[0, :] = mu.item(0), mu.item(2)

# Measurements
delta = 0.5  # 3125
z = C @ X + np.array([[1.], [1.]]) * np.random.randn() * delta  # First measurement
z_log = np.zeros([size, 2])
z_log[0, :] = z.item(0), z.item(1)
R = np.eye(2) * delta  # Measurement uncertainty

filter = Kf(mu, SIG, u, z, A, B, C, R, Q)

for iter in range(1, size):

    # Compute truth
    X = A @ X_1 + B @ u
    truth[iter, 0] = X.item(0)
    X_1 = X
    if X.item(2) > 0.:
        truth[iter, 1] = X.item(2)
        X_1 = X
    else:
        break

    # Collect measurement
    z = C @ X + np.array([[1.], [1.]]) * np.random.randn() * delta
    z_log[iter, :] = z.item(0), z.item(1)

    mu, SIG = filter.update(X, SIG, u, z)
    mu_log[iter, :] = mu.item(0), mu.item(2)

plt.plot(truth[:iter, 0], truth[:iter, 1])
plt.axis([-1, np.amax(z_log) + 1, -1, np.amax(z_log) / 2.])
plt.plot(mu_log[:iter, 0], mu_log[:iter, 1])
plt.plot(z_log[:iter, 0], z_log[:iter, 1])
plt.show()
