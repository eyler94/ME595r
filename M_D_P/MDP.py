#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Mdp:
    def __init__(self):
        self.N = 100
        self.Np = self.N + 2
        self.gen_map()
        self.gen_cost_map()
        self.X = np.arange(1, self.N+1, 1)
        self.Y = np.arange(1, self.N+1, 1)
        self.u = np.zeros([self.N, self.N])
        self.v = np.zeros([self.N, self.N])

    def gen_map(self):
        map = np.zeros([self.Np, self.Np])  # map dimension

        # Initialize walls and obstacle maps as empty
        self.walls = np.zeros([self.Np, self.Np])
        self.obs1 = np.zeros([self.Np, self.Np])
        self.obs2 = np.zeros([self.Np, self.Np])
        self.obs3 = np.zeros([self.Np, self.Np])
        self.goal = np.zeros([self.Np, self.Np])

        # Create exterior walls
        self.walls[0:2, 0:self.N + 1] = 1
        self.walls[0:self.N + 1, 0:2] = 1
        self.walls[-2:self.Np, 0:self.N + 1] = 1
        self.walls[0:self.Np, -2:self.Np] = 1

        # Create single obstacle
        self.obs1[19:41, 29:79] = 1
        self.obs1[9:19, 59:64] = 1

        # Another obstacle
        self.obs2[44:64, 9:44] = 1

        # Another obstacle
        self.obs3[42:91, 74:84] = 1
        self.obs3[69:79, 49:74] = 1

        # The goal states
        self.goal[74:79, 95:97] = 1

        # Put walls and obstacles into map
        self.map_orig = self.walls + self.obs1 + self.obs2 + self.obs3 + self.goal
        self.map = self.map_orig.T
        self.map_inv = np.logical_not(self.map[1:-1, 1:-1])
        # Plot map
        fig1 = plt.figure(1)
        fig1.clf()
        plt.imshow(self.map, origin='lower')  # , "Greys")

        fig2 = plt.figure(2)
        fig2.clf()
        plt.imshow(self.map_orig, origin='lower')  # , "Greys")
        plt.draw()
        plt.pause(0.001)
        self.obst_cost = -5000
        self.wall_cost = -100
        self.goal_reward = 100000
        self.cost = -2

    def gen_cost_map(self):
        self.walls_cost = self.walls * self.wall_cost
        self.obs1_cost = self.obs1 * self.obst_cost
        self.obs2_cost = self.obs2 * self.obst_cost
        self.obs3_cost = self.obs3 * self.obst_cost
        self.goal_cost = self.goal * self.goal_reward
        self.cost_map_orig = self.walls_cost + self.obs1_cost + self.obs2_cost + self.obs3_cost + self.goal_cost
        # self.cost_map_orig[self.cost_map_orig != 0] -= 2
        self.cost_map = self.cost_map_orig.T
        self.inner = self.cost_map[1:-1, 1:-1]

    def MDP_discrete_value_iteration(self, LivePlotting=True):
        self.V = deepcopy(self.cost_map)
        self.V_1 = deepcopy(self.cost_map) - 10
        iteration = 0
        self.tesseract = np.zeros([4, self.N, self.N])
        while np.abs(np.sum(self.V - self.V_1)) >= 1e-2 and iteration < 10000:
            iteration += 1
            N = self.V[2:, 1:-1]
            E = self.V[1:-1, 2:]
            S = self.V[:-2, 1:-1]
            W = self.V[1:-1, :-2]
            self.tesseract[0] = self.cost + 0.8 * N + 0.1 * W + 0.1 * E  # facing north
            self.tesseract[1] = self.cost + 0.8 * E + 0.1 * N + 0.1 * S  # facing north
            self.tesseract[2] = self.cost + 0.8 * S + 0.1 * E + 0.1 * W  # facing north
            self.tesseract[3] = self.cost + 0.8 * W + 0.1 * S + 0.1 * N  # facing north
            self.tesseract[:, self.inner != 0] = self.inner[self.inner != 0]
            self.V_1 = deepcopy(self.V)
            self.V[1:-1, 1:-1] = np.max(self.tesseract, 0)
            self.policy = np.argmax(self.tesseract, 0)

            # Plotting
            if LivePlotting:
                print("iter", iteration)
                fig3 = plt.figure(3)
                fig3.clf()
                image3 = plt.imshow(self.V, origin='lower')  # , "Greys")
                fig3.colorbar(image3)
                self.plot_arrows()

                plt.draw()
                plt.pause(0.00001)

    def Algorithm_policy_MDP(self):
        self.policy = np.argmax(self.tesseract, 0)

    def plot_arrows(self):
        self.u[self.policy == 0] = 0
        self.u[self.policy == 1] = 1
        self.u[self.policy == 2] = 0
        self.u[self.policy == 3] = -1

        self.v[self.policy == 0] = 1
        self.v[self.policy == 1] = 0
        self.v[self.policy == 2] = -1
        self.v[self.policy == 3] = 0

        # plt.quiver(self.X, self.Y, self.u, self.v)
        plt.quiver(self.X, self.Y, self.u*self.map_inv, self.v*self.map_inv)

