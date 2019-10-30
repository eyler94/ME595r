#!/usr/bin/env python3
# Class defining a constant altitude quadrotor. It has a sensor that provides range and bearing to landmarks
# which it uses to navigate around the world.

import numpy as np

pi = np.pi


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


class quadrotor:
    def __init__(self, x0=-5., y0=0., theta0=pi / 2., ts=0.1, tf=30.):
        # Initial Conditions
        self.x0 = x0  # m
        self.y0 = y0  # m
        self.theta0 = theta0  # rad

        # Time parameters
        self.ts = ts  # s
        self.t0 = 0.  # s
        self.tf = tf  # s

        # Noise parameters
        self.sig_v = 0.15
        self.sig_omega = 0.1

        self.sigma_r = 0.2  # m
        self.sigma_theta = 0.1  # rad

        self.x = self.x0
        self.y = self.y0
        self.theta = self.theta0

        self.v_t = 0
        self.omega_t = 0

    def propagate_dynamics(self, v, omega):
        self.update_velocity(v, omega)

        self.x = self.x + self.v_t * np.cos(self.theta) * self.ts
        self.y = self.y + self.v_t * np.sin(self.theta) * self.ts
        self.theta = wrapper(self.theta + self.omega_t * self.ts)

        return self.x, self.y, self.theta

    def update_velocity(self, v, omega):
        self.v_t = v + np.random.randn()*0.15
        self.omega_t = omega + np.random.randn()*0.1

    # def calculate_measurements(self, num_landmarks, landmarks):
    #     # print("Calculating measurements.")
    #     R = np.zeros([num_landmarks, 1])
    #     PH = np.zeros([num_landmarks, 1])
    #
    #     for iter in range(0, num_landmarks):
    #         # print("Looping through landmarks.")
    #         x = landmarks[0][iter] - self.x
    #         y = landmarks[1][iter] - self.y
    #         R[iter] = np.sqrt(x ** 2 + y ** 2) + np.random.randn() * self.sigma_r
    #         PH[iter] = wrapper(np.arctan2(y, x) - self.theta + np.random.randn() * self.sigma_theta)
    #     return R, PH
