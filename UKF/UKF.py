#!/usr/bin/env python3

import numpy as np
import math
from scipy.linalg import block_diag
from numpy.linalg import cholesky as ch

import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


class UKF:
    def __init__(self, R2D2, World):
        self.mu = np.array([[R2D2.x0],
                            [R2D2.y0],
                            [R2D2.theta0]])
        self.SIG = np.diag([0.1, 0.1, 0.1])
        self.v = 0
        self.omega = 1
        meas = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
        self.r = meas[0]
        self.ph = meas[1]
        self.u = np.array([[self.v],
                           [self.omega]])
        self.current_landmark = 0
        self.landmarks = World.Landmarks

        # Generate augmented mean and covariance
        self.ts = R2D2.ts
        self.alpha1 = R2D2.alpha1
        self.alpha2 = R2D2.alpha2
        self.alpha3 = R2D2.alpha3
        self.alpha4 = R2D2.alpha4
        self.M = np.array([[self.alpha1 * self.v ** 2 + self.alpha2 * self.omega ** 2, 0],
                           [0, self.alpha3 * self.v ** 2 + self.alpha4 * self.omega ** 2]])
        self.Q = np.array([[R2D2.sigma_r ** 2, 0],
                           [0, R2D2.sigma_theta ** 2]])
        self.mu_a = np.hstack([self.mu.T, np.zeros([1, 4])])
        self.mu_a = np.reshape(self.mu_a, [7, 1])
        self.mu_a = np.reshape(self.mu_a, [7, 1])
        self.SIG_a = block_diag(self.SIG, self.M, self.Q)

        # Generate Sigma points
        self.alpha = 0.4
        self.kappa = 4
        self.beta = 2
        self.n = 7
        self.lamb_duh = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lamb_duh)
        self.Chi_a = np.hstack(
            [self.mu_a, self.mu_a + self.gamma * ch(self.SIG_a), self.mu_a - self.gamma * ch(self.SIG_a)])
        self.Chi_a[[2, 6], :] = wrapper(self.Chi_a[[2, 6], :])
        self.Chi_x = self.Chi_a[0:3, :]
        self.Chi_u = self.Chi_a[3:5, :]
        self.Chi_z = self.Chi_a[5:, :]

        # Pass sigma points through motion model and compute Gaussian statistics
        self.Chi_bar = self.g(self.u + self.Chi_u, self.Chi_x)

        # Calculate weights
        self.w_m = np.ones([1, 15])
        self.w_c = np.ones([1, 15])
        self.w_m[0] = self.lamb_duh / (self.n + self.lamb_duh)
        self.w_c[0] = self.w_m[0] + (1 - self.alpha ** 2 + self.beta)
        for spot in range(1, 15):
            self.w_m[0][spot] = 1 / (2 * (self.n + self.lamb_duh))
            self.w_c[0][spot] = 1 / (2 * (self.n + self.lamb_duh))

        self.mu_bar = self.Chi_x @ self.w_m.T
        self.SIG_bar = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Chi_bar - self.mu_bar).T

        # Predict observations at sigma points and compute Gaussian statistics
        self.Z_bar = self.h(self.Chi_bar, self.landmarks[:, self.current_landmark]) + self.Chi_z
        self.z_hat = self.Z_bar @ self.w_m.T
        self.S = np.multiply(self.w_c, (self.Z_bar - self.z_hat)) @ (self.Z_bar - self.z_hat).T
        self.SIG_xz = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Z_bar - self.z_hat).T

        # Update mean and covariance
        self.K = self.SIG_xz @ np.linalg.inv(self.S)
        # z_diff = np.array([self.r[self.current_landmark]-self.z_hat[0],
        #                    wrapper(self.ph[self.current_landmark]-self.z_hat[1])])
        # self.mu = self.mu_bar + self.K @ z_diff
        # self.SIG = self.SIG_bar - self.K @ self.S @ self.K.T
        # self.current_landmark = 1
        # self.lines4_16_wo_7()
        # self.current_landmark = 2
        # self.lines4_16_wo_7()

    def g(self, u, state):
        v = u[0]
        omega = u[1]

        x = state[0]
        y = state[1]
        theta = state[2]

        x = x - v / omega * np.sin(theta) + v / omega * np.sin(wrapper(theta + omega * self.ts))
        y = y + v / omega * np.cos(theta) - v / omega * np.cos(wrapper(theta + omega * self.ts))
        theta = wrapper(theta + omega * self.ts)

        return x, y, theta

    def h(self, state, landmark):
        x = landmark[0] - state[0]
        y = landmark[1] - state[1]

        r = np.sqrt(x ** 2 + y ** 2)
        ph = wrapper(np.arctan2(y, x) - state[2])

        return r, ph

    def lines4_16_wo_7(self):
        # Generate augmented mean and covariance
        self.mu_a = np.hstack([self.mu.T, np.zeros([1, 4])])
        self.mu_a = np.reshape(self.mu_a, [7, 1])
        self.SIG_a = block_diag(self.SIG, self.M, self.Q)
        # Generate Sigma points
        self.Chi_a = np.hstack(
            [self.mu_a, self.mu_a + self.gamma * ch(self.SIG_a), self.mu_a - self.gamma * ch(self.SIG_a)])
        self.Chi_a[[2, 6], :] = wrapper(self.Chi_a[[2, 6], :])
        self.Chi_x = self.Chi_a[0:3, :]
        self.Chi_u = self.Chi_a[3:5, :]
        self.Chi_z = self.Chi_a[5:, :]
        self.Chi_bar = self.Chi_x
        self.mu_bar = self.Chi_x @ self.w_m.T
        self.SIG_bar = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Chi_bar - self.mu_bar).T
        # Predict observations at sigma points and compute Gaussian statistics
        self.Z_bar = self.h(self.Chi_bar, self.landmarks[:, self.current_landmark]) + self.Chi_z
        self.z_hat = self.Z_bar @ self.w_m.T
        self.S = np.multiply(self.w_c, (self.Z_bar - self.z_hat)) @ (self.Z_bar - self.z_hat).T
        self.SIG_xz = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Z_bar - self.z_hat).T
        # Update mean and covariance
        self.K = self.SIG_xz @ np.linalg.inv(self.S)
        z_diff = np.array([self.r[self.current_landmark] - self.z_hat[0],
                           wrapper(self.ph[self.current_landmark] - self.z_hat[1])])
        self.mu = self.mu_bar + self.K @ z_diff
        self.SIG = self.SIG_bar - self.K @ self.S @ self.K.T

    def update(self, mu, SIG, v, omega, r, ph):
        self.mu = mu
        self.SIG = SIG
        self.v = v
        self.omega = omega
        self.r = r
        self.ph = ph
        self.u = np.array([[self.v],
                           [self.omega]])
        self.current_landmark = 0
        self.M = np.array([[self.alpha1 * self.v ** 2 + self.alpha2 * self.omega ** 2, 0],
                           [0, self.alpha3 * self.v ** 2 + self.alpha4 * self.omega ** 2]])
        self.mu_a = np.hstack([self.mu.T, np.zeros([1, 4])])
        self.mu_a = np.reshape(self.mu_a, [7, 1])
        self.SIG_a = block_diag(self.SIG, self.M, self.Q)

        # Generate Sigma points
        self.alpha = 0.4
        self.kappa = 4
        self.beta = 2
        self.n = 7
        self.lamb_duh = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lamb_duh)
        self.Chi_a = np.hstack(
            [self.mu_a, self.mu_a + self.gamma * ch(self.SIG_a), self.mu_a - self.gamma * ch(self.SIG_a)])
        self.Chi_a[[2, 6], :] = wrapper(self.Chi_a[[2, 6], :])
        self.Chi_x = self.Chi_a[0:3, :]
        self.Chi_u = self.Chi_a[3:5, :]
        self.Chi_z = self.Chi_a[5:, :]

        # Pass sigma points through motion model and compute Gaussian statistics
        self.Chi_bar = self.g(self.u + self.Chi_u, self.Chi_x)
        self.mu_bar = self.Chi_x @ self.w_m.T
        self.SIG_bar = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Chi_bar - self.mu_bar).T

        # Predict observations at sigma points and compute Gaussian statistics
        self.Z_bar = self.h(self.Chi_bar, self.landmarks[:, self.current_landmark]) + self.Chi_z
        self.z_hat = self.Z_bar @ self.w_m.T
        self.S = np.multiply(self.w_c, (self.Z_bar - self.z_hat)) @ (self.Z_bar - self.z_hat).T
        self.SIG_xz = np.multiply(self.w_c, (self.Chi_bar - self.mu_bar)) @ (self.Z_bar - self.z_hat).T

        # Update mean and covariance
        self.K = self.SIG_xz @ np.linalg.inv(self.S)
        z_diff = np.array([self.r[self.current_landmark] - self.z_hat[0],
                           wrapper(self.ph[self.current_landmark] - self.z_hat[1])])
        self.mu = self.mu_bar + self.K @ z_diff
        self.SIG = self.SIG_bar - self.K @ self.S @ self.K.T
        self.current_landmark = 1
        self.lines4_16_wo_7()
        self.current_landmark = 2
        self.lines4_16_wo_7()
        return self.mu, self.SIG
