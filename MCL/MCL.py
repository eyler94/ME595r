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


class MCL:
    def __init__(self, R2D2, World):
        print("Initializing Particle Filter.")
        # World Properties
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
        x_p = np.random.uniform(-10, 10, [1, self.num_particles])
        y_p = np.random.uniform(-10, 10, [1, self.num_particles])
        th_p = np.random.uniform(-np.pi, np.pi, [1, self.num_particles])
        # particles = np.vstack([x_p, y_p, th_p])  # +state0
        particles = np.zeros([3, 10]) + state0
        self.particles = particles

    def update(self, v, omega, r, ph):  # Based on Table 8.2
        print("Updating particle filter.")
        print("Sample Motion Model to generate new particle.")
        u = np.array([[v],
                      [omega]])
        particles = self.g(u, self.particles)
        # particles = np.array([[particles[0]],
        #                       [particles[1]],
        #                       [particles[2]]])
        print("Measurement Model to calculate weight for that particle.")  # table 6.4+(-theta) and pg 123
        weights = self.weight(r, ph, particles)
        print("Append it to the temp particle set.")
        Xtbar = np.vstack([particles, weights])
        print("Redraw particles for the real particle set from the temp particle set according to their weight.")
        self.particles = self.low_variance_sampler(Xtbar, weights)
        print("Calculate the mean of x, y, and theta.")
        print("Calculate the standard deviation of x, y, and theta.")

    def g(self, u, state):
        v = u[0]
        omega = u[1]

        x = state[0]
        y = state[1]
        theta = state[2]

        x = x - v / omega * np.sin(theta) + v / omega * np.sin(wrapper(theta + omega * self.ts))
        y = y + v / omega * np.cos(theta) - v / omega * np.cos(wrapper(theta + omega * self.ts))
        theta = wrapper(theta + omega * self.ts)

        return np.array([[x],
                         [y],
                         [theta]])

    def weight(self, r, ph, Xtbar):  # Based on Table 6.4
        print("Generating weights.")
        P = 1
        for lm in range(0, self.num_landmarks):
            x = self.landmarks[0][lm] - Xtbar[0]
            y = self.landmarks[1][lm] - Xtbar[1]
            theta = Xtbar[2]
            r_hat = np.sqrt(x ** 2 + y ** 2)
            phi_hat = wrapper(np.arctan(y, x) - theta)
            P *= prob(r[lm] - r_hat, self.sigma_r) * prob(wrapper(ph[lm] - phi_hat), self.sigma_theta)
        # if P.any() < 0.001:
        #     P = 0.001
        return P

    def low_variance_sampler(self, Xt, Wt):  # Based on Table 4.4
        # How do we deal with the 2d shape of the particles in this?
        print("Low variance sampler.")
        Xtbar = np.array([]).reshape([3, 0])
        r = np.random.rand() * self.M_inv
        c = Wt[0]
        i = 1
        for m in range(self.num_particles):
            U = r + m * self.M_inv
            while U > c:
                # print("U:", U)
                # print("c:", c)
                i = i + 1
                if i == self.num_particles:
                    print("Failed to find high enough weight.")
                c = c + Wt[i]
            Xtbar = np.hstack([Xtbar, Xt[:][i]])
        return Xtbar
