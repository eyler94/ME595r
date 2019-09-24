#!/usr/bin/env python3

# Class defining a two wheeled autonomous robot. It has a sensor that provides range and bearing to landmarks
# which it uses to navigate around the world.

import numpy as np

pi = np.pi


class R2D2:
    def __init__(self, x0=-5., y0=-3., theta0=pi / 2., ts=0.1, tf=20.):
        # Initial Conditions
        self.x0 = x0  # m
        self.y0 = y0  # m
        self.theta0 = theta0  # rad

        # Time parameters
        self.ts = ts  # s
        self.t0 = 0.  # s
        self.tf = tf  # s

        # Noise parameters
        self.alpha1 = 0.1
        self.alpha2 = 0.01
        self.alpha3 = 0.01
        self.alpha4 = 0.1

        self.sigma_r = 0.1  # m
        self.sigma_theta = 0.05  # rad

        # Velocity models
        t = self.t0
        self.v_c = 1 + 0.5 * np.cos(2 * pi * 0.2 * t)
        self.omega_c = -0.2 + 2 * np.cos(2 * pi * 0.6 * t)

        # States
        # self.x_1 = self.x0
        # self.y_1 = self.y0
        # self.theta_1 = self.theta0

        self.x = self.x0
        self.y = self.y0
        self.theta = self.theta0

    def propagateDynamics(self, ts):
        v_hat = self.v_c + np.random.rand()*np.sqrt(self.alpha1*self.v_c**2 + self.alpha2*self.omega_c**2)
        omega_hat = self.omega_c + np.random.rand()*np.sqrt(self.alpha3*self.v_c**2 + self.alpha4*self.omega_c**2)

        self.x = self.x - v_hat / omega_hat * np.sin(self.theta) + v_hat / omega_hat * np.sin(self.theta + omega_hat * ts)
        self.y = self.y + v_hat / omega_hat * np.cos(self.theta) - v_hat / omega_hat * np.cos(self.theta + omega_hat * ts)
        self.theta = self.theta + omega_hat*ts

        return self.x, self.y, self.theta

        print("Propagate dynamics.")


