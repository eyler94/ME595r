#!/usr/bin/env python3

import numpy as np
import scipy.io as spio
from importlib import reload
from time import sleep
import PlotterOGM as PLTTER
import OGM as OG

reload(PLTTER)
reload(OG)

# Load provided data from mat file
mat = spio.loadmat('state_meas_data.mat')
state = mat['X']
meas = mat['z']
angles = mat['thk']

# # Inverse Sensor Range Model Parameters
# alpha = 1  # wall thickness in meters
# beta = 5  # degrees
# z_max = 150  # meters

# # Probability Parameters
# p_hit_high = 0.7
# p_hit_low = 0.6
# p_no_hit_high = 0.4
# p_no_hit_low = 0.3

# World Parameters
map_Height = 100  # meters
map_Width = 100  # meters


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


# Instantiate Plotter and OGM
Plotter = PLTTER.Plotter(state[0, 0], state[1, 0], state[2, 0], map_Width, map_Height)
ogm = OG.OGM()

for spot in range(0, state.shape[1]):
    ogm.update(state[:, spot], meas[:, :, spot])
    Plotter.update(state[0, spot],
                   state[1, spot],
                   state[2, spot],
                   ogm.info)

input()