#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt



N = 100
Np = 100 + 2

map = np.zeros([Np, Np])  # map dimension

# Initialize walls and obstacle maps as empty
walls = np.zeros([Np, Np])
obs1 = np.zeros([Np, Np])
obs2 = np.zeros([Np, Np])
obs3 = np.zeros([Np, Np])
goal = np.zeros([Np, Np])

# Create exterior walls
walls[1, 1:N] = 1
walls[1:N + 1, 1] = 1
walls[N, 1:N + 1] = 1
walls[1:N + 1, N] = 1

# Create single obstacle
obs1[19:39, 29:79] = 1
obs1[9:19, 59:64] = 1

# Another obstacle
obs2[44:64, 9:44] = 1

# Another obstacle
obs3[42:91, 74:84] = 1
obs3[69:79, 49:74] = 1

# The goal states
goal[74:79, 95:97] = 1

# Put walls and obstacles into map
map = walls + obs1 + obs2 + obs3 + goal
# # Plot map
fig1 = plt.figure(1)
fig1.clf()

plt.imshow(np.flipud(map.T))#, "Greys")
plt.show()


