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
        self.num_particles = 1000
        self.M_inv = 1 / self.num_particles
        self.particles = np.zeros([2, self.num_particles])
        self.particles_t_1 = np.zeros([2, self.num_particles])
        self.u = np.array([[R2D2.v_c],
                           [R2D2.omega_c]])
        self.z = R2D2.calculate_measurements(World.Number_Landmarks,World.Landmarks)
        self.ts = R2D2.ts

    def update(self, particles_t_1, u, z):  # table 8.2
        print("Updating particle filter.")
        print("Initialize empty temp particle set.")
        print("Sample Motion Model to generate new particle.")
        print("Measurement Model to calculate weight for that particle.")  # table 6.4+(-theta) and pg 123
        print("Append it to the temp particle set.")
        print("Draw particles for the real particle set according to their weight.")
        # self.particles = particles
        # self.mu = mu
        # self.SIG = SIG

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
        ph = np.arctan2(y, x) - state[2]

        return r, ph

    def low_variance_sampler(self, Xt, Wt): ##### How do we deal with the 2d shape of the particles in this?
        Xtbar = np.array([]).reshape([0, 0])
        r = np.random.rand() * self.M_inv
        c = Wt[0][0]
        i = 1
        for m in range(1, self.num_particles):
            U = r + (m - 1) * self.M_inv
            while U > c:
                i = i + 1
                c = c + Wt[0][i]
            Xtbar = np.hstack([Xtbar,Xt[0][i]])
        return Xtbar
