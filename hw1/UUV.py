#!/usr/bin/env python3

import numpy as np
from control import matlab as mt


class UUV():
    def __init__(self, m=100., b=20., F=50., ts=0.05, tf=50):
        # sys parameters
        self.m = m
        self.b = b
        self.F_norm = F

        # Time parameters
        self.ts = ts
        self.t0 = 0
        self.t1 = 5
        self.t2 = 25
        self.t3 = 30
        self.tf = tf

        # Noise parameters
        self.sigma_pos_meas = 0.001  # m^2
        self.Q = self.sigma_pos_meas
        self.sigma_vel = 0.01  # m^2/s^2
        self.sigma_pos = 0.0001  # m^2
        self.R = np.diagflat([self.sigma_pos, self.sigma_vel])

        self.mu = np.array([[0.],
                            [0.]])
        self.SIG = self.R

        # SS parameters
        self.F = np.array([[0., 1.],
                           [0., -self.b / self.m]])
        self.G = np.array([[0.],
                           [1. / self.m]])
        self.H = np.array([1., 0.])
        self.J = np.array([0.])
        self.sysd = mt.ss(self.F, self.G, self.H, self.J, self.ts)
        self.A = self.sysd.A
        self.B = self.sysd.B
        self.C = self.sysd.C
        self.D = self.sysd.D

        # States
        self.X_1 = np.array([[0.],
                             [0.]])

        self.X = np.array([[0.],
                           [0.]])

    def propagate_dynamics(self, u):
        self.X = self.A @ self.X_1 + self.B @ u + process_noise
        return self.X

    def collect_measurements(self):
        z = self.C @ self.X + measurement_noise
        return z
