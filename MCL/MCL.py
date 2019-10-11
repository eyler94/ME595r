#!/usr/bin/env python3

import numpy as np
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


def prob(a, b_sqrd):
    return 1 / np.sqrt(2 * np.pi * b_sqrd) * np.exp(-0.5 * a ** 2 / b_sqrd)


DEBUG = False


def printer(statement):
    if DEBUG:
        print(statement)


class MCL:
    def __init__(self, R2D2, World):
        printer("Initializing Particle Filter.")
        # World Properties
        self.alpha1 = R2D2.alpha1
        self.alpha2 = R2D2.alpha2
        self.alpha3 = R2D2.alpha3
        self.alpha4 = R2D2.alpha4
        self.alpha5 = R2D2.alpha5
        self.alpha6 = R2D2.alpha6
        self.u = np.array([[R2D2.v_c],
                           [R2D2.omega_c]])
        self.z = R2D2.calculate_measurements(World.Number_Landmarks, World.Landmarks)
        self.ts = R2D2.ts
        self.num_landmarks = World.Number_Landmarks
        self.landmarks = World.Landmarks
        self.sigma_r = R2D2.sigma_r
        self.sigma_theta = R2D2.sigma_theta

        # Filter Properties
        self.num_particles = 1000
        self.M_inv = 1 / self.num_particles
        state0 = np.array([[R2D2.x0],
                           [R2D2.y0],
                           [R2D2.theta0]])
        dist = 10
        x_p = np.random.uniform(-dist, dist, [1, self.num_particles])
        y_p = np.random.uniform(-dist, dist, [1, self.num_particles])
        th_p = np.random.uniform(-np.pi, np.pi, [1, self.num_particles])
        particles = np.vstack([x_p, y_p, th_p])  # + state0
        # particles = np.zeros([3, self.num_particles]) + state0
        self.particles = particles

        # Filter Stats
        self.mu_x = self.particles[0].mean()
        self.mu_y = self.particles[1].mean()
        self.mu_th = self.particles[2].mean()
        self.sig_x = self.particles[0].std()
        self.sig_y = self.particles[1].std()
        self.sig_th = self.particles[2].std()

    def update(self, v, omega, r, ph):  # Based on Table 8.2
        printer("Updating particle filter.")
        printer("Sample Motion Model to generate new particle.")
        u = np.array([[v],
                      [omega]])
        self.particles = self.g(u, self.particles)
        self.particles = np.vstack([self.particles[0], self.particles[1], self.particles[2]])
        printer("Measurement Model to calculate weight for that particle.")  # table 6.4+(-theta) and pg 123
        weights = self.weight(r, ph, self.particles)
        printer("Append it to the temp particle set.")
        Xtbar = np.vstack([self.particles, weights])
        printer("Redraw particles for the real particle set from the temp particle set according to their weight.")
        self.particles = self.low_variance_sampler(Xtbar, weights)
        printer("Calculate the mean of x, y, and theta.")
        self.mu_x = self.particles[0].mean()
        self.mu_y = self.particles[1].mean()
        self.mu_th = self.particles[2].mean()
        printer("Calculate the standard deviation of x, y, and theta.")
        self.sig_x = self.particles[0].std()
        self.sig_y = self.particles[1].std()
        self.sig_th = self.particles[2].std()

    def g(self, u, state):
        v = u[0]
        omega = u[1]

        v_hat = v + np.random.randn(1, self.num_particles) * np.sqrt(self.alpha1 * v ** 2 + self.alpha2 * omega ** 2)
        omega_hat = omega + np.random.randn(1, self.num_particles) * np.sqrt(
            self.alpha3 * v ** 2 + self.alpha4 * omega ** 2)
        gamma_hat = np.random.randn(1, self.num_particles) * np.sqrt(self.alpha5 * v ** 2 + self.alpha6 * omega ** 2)

        x = state[0]
        y = state[1]
        theta = state[2]

        x = x - v_hat / omega_hat * np.sin(theta) + v_hat / omega_hat * np.sin(wrapper(theta + omega_hat * self.ts))
        y = y + v_hat / omega_hat * np.cos(theta) - v_hat / omega_hat * np.cos(wrapper(theta + omega_hat * self.ts))
        theta = wrapper(theta + omega_hat * self.ts + gamma_hat * self.ts)

        return x, y, theta

    def weight(self, r, ph, Xtbar):  # Based on Table 6.4
        printer("Generating weights.")
        P = 1
        for lm in range(0, self.num_landmarks):
            x = self.landmarks[0][lm] - Xtbar[0]
            y = self.landmarks[1][lm] - Xtbar[1]
            theta = Xtbar[2]
            r_hat = np.sqrt(x ** 2 + y ** 2)
            phi_hat = wrapper(np.arctan2(y, x) - theta)
            P *= prob(r[lm] - r_hat, self.sigma_r**2) * prob(wrapper(ph[lm] - phi_hat), self.sigma_theta**2)
        P = P / np.sum(P)
        return P

    def low_variance_sampler(self, Xt, Wt):  # Based on Table 4.4
        printer("Low variance sampler.")
        Xtbar = np.array([]).reshape([4, 0])
        r = np.random.rand() * self.M_inv
        c = Wt[0]
        i = 0
        for m in range(self.num_particles):
            U = r + m * self.M_inv
            while U > c:
                i = i + 1
                c = c + Wt[i]
            xti = Xt[:, i].reshape([4, 1])
            Xtbar = np.hstack([Xtbar, xti])
        return Xtbar
