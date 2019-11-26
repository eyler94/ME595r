#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append('..')
from M_D_P import MDP

from importlib import reload

reload(MDP)

import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Instantiate World, Robot, Plotter, and ekfslam
mdp = MDP.Mdp()

mdp.MDP_discrete_value_iteration(LivePlotting=True)

mdp.Algorithm_policy_MDP()

print("done")
# Plotting
fig1 = plt.figure(1)
fig1.clf()
image1 = plt.imshow(-mdp.map, origin='lower')  # , "Greys")
fig1.colorbar(image1)

fig2 = plt.figure(2)
fig2.clf()
image2 = plt.imshow(-mdp.cost_map, origin='lower')  # , "Greys")
fig2.colorbar(image2)

fig3 = plt.figure(3)
fig3.clf()
image3 = plt.imshow(mdp.V, origin='lower')  # , "Greys")
mdp.plot_arrows()
#
# X = np.arange(0, 100, 1)
# Y = np.arange(0, 100, 1)
# U = np.zeros([100, 100])
# V = np.zeros([100, 100])
# U[mdp.policy==0] = 0
# U[mdp.policy==1] = 1
# U[mdp.policy==2] = 0
# U[mdp.policy==3] = -1
#
# V[mdp.policy==0] = 1
# V[mdp.policy==1] = 0
# V[mdp.policy==2] = -1
# V[mdp.policy==3] = 0
#
# plt.quiver(X, Y, U, V)
# #
# # fig, ax = plt.subplots()
# # q = ax.quiver(X, Y, U, V)

plt.show()
