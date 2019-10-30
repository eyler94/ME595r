#!/usr/bin/env python3

import numpy as np
import scipy.io as spio
from importlib import reload
from time import sleep
import sys

sys.path.append('..')
from Midterm import Quad # import quadrotor
from Midterm import QWorld # import World
from Plotter import Plotter
from EKF import EKF
from importlib import reload

reload(Quad)
reload(QWorld)
reload(Plotter)
reload(EKF)


def wrapper(ang):
    ang -= np.pi*2 * np.floor((ang + np.pi) / (2*np.pi))
    return ang


mat = spio.loadmat('midterm_data.mat')
state = mat['X']
omega_commanded = mat['om_c']
omega_actual = mat['om']
velocity_actual = mat['v']

print("bah")

