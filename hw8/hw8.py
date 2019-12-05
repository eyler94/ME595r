#!/usr/bin/env python3

import matplotlib.pyplot as plt
from importlib import reload
import sys
sys.path.append('..')
from M_D_P import MDP


reload(MDP)

# Instantiate World, Robot, Plotter, and ekfslam
mdp = MDP.Mdp()

mdp.MDP_discrete_value_iteration(LivePlotting=True)

mdp.Algorithm_policy_MDP()

# Robot initial location
x = 60
y = 60

mdp.plot_path(x, y, LivePlotting=True)

# Plotting
fig3 = plt.figure(3)
fig3.clf()
image3 = plt.imshow(mdp.V, origin='lower')  # , "Greys")
mdp.plot_arrows()

mdp.live_plotter(Path=mdp.path)
plt.show()
