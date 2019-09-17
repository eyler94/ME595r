#!/usr/bin/env python3

import numpy as np
import control as ctl
from control import matlab as mt


class UUV():
    def __init__(self, m=100., b=20., ts=0.05, tf=50):
        # sys parameters
        self.m = m
        self.b = b

        # Time parameters
        self.ts = ts
        self.t0 = 0
        self.t1 = 5
        self.t2 = 25
        self.t3 = 30
        self.tf = tf

        # Noise parameters
        self.sigma_meas = 0.001  # m^2
        self.sigma_vel = 0.01  # m^2/s^2
        self.sigma_pos = 0.0001  # m^2

        # SS parameters
        self.F = np.array([[0., 1.],
                           [0., -self.b / self.m]])
        self.G = np.array([[0.],
                           [1. / self.m]])
        self.H = np.array([1., 0.])
        self.J = np.array([0.])
        self.sysd = mt.ss(self.F, self.G, self.H, self.J, self.Ts)
