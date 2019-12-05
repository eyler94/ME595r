#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Mdp:
    def __init__(self):
        self.N = 100  # Nominal length of one side
        self.Np = self.N + 2  # Length accounting for size of array
        self.gen_map()  # Generate the map of obstacles using the class's function.
        self.gen_cost_map()  # Generate the cost based on the given constraints

        # Data necessary for plotting the vector field
        self.X = np.arange(1, self.N + 1, 1)
        self.Y = np.arange(1, self.N + 1, 1)
        self.u = np.zeros([self.N, self.N])
        self.v = np.zeros([self.N, self.N])

    def gen_map(self):
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
        self.obs2[44:70, 9:49] = 1

        # Another obstacle
        self.obs3[42:91, 74:84] = 1
        self.obs3[69:79, 49:74] = 1

        # The goal states
        self.goal_x_low = 74
        self.goal_x_high = 79
        self.goal_y_low = 95
        self.goal_y_high = 97
        self.goal[self.goal_x_low:self.goal_x_high, self.goal_y_low:self.goal_y_high] = 1

        # Put walls and obstacles into map
        self.map_orig = self.walls + self.obs1 + self.obs2 + self.obs3 + self.goal
        self.map = self.map_orig.T
        self.map_inv = np.logical_not(self.map[1:-1, 1:-1])
        # # Plot map
        # fig1 = plt.figure(1)
        # fig1.clf()
        # plt.imshow(self.map, origin='lower')  # , "Greys")
        #
        # fig2 = plt.figure(2)
        # fig2.clf()
        # plt.imshow(self.map_orig, origin='lower')  # , "Greys")
        # plt.draw()
        # plt.pause(0.001)

    def gen_cost_map(self):
        # Cost associated with obstacles, walls, movement, and the goal.
        self.obst_cost = -5000
        self.wall_cost = -100
        self.movement_cost = -2
        self.goal_reward = 100000

        # Generating a map of the cost associated with the terrain.
        self.walls_cost = self.walls * self.wall_cost
        self.obs1_cost = self.obs1 * self.obst_cost
        self.obs2_cost = self.obs2 * self.obst_cost
        self.obs3_cost = self.obs3 * self.obst_cost
        self.goal_cost = self.goal * self.goal_reward
        self.cost_map_orig = self.walls_cost + self.obs1_cost + self.obs2_cost + self.obs3_cost + self.goal_cost
        self.cost_map = self.cost_map_orig.T

        # A map of just the interior (not including the walls.
        self.inner = self.cost_map[1:-1, 1:-1]

    def MDP_discrete_value_iteration(self, LivePlotting=True):
        self.V = deepcopy(self.cost_map)  # Initialize current map associated with terrain and movement.
        self.V_1 = deepcopy(self.cost_map) - 10  # Initialize previous map associated with terrain and movement.
        self.iteration = 0  # Start the iteration counter.
        self.tesseract = np.zeros([4, self.N, self.N])  # Tensor designated for holding the costs associated with
        # movement given the orientation
        while np.abs(np.sum(self.V - self.V_1)) >= 1e-2 and self.iteration < 10000:  # If the difference between runs
            # drops or it takes too long, quit.
            self.iteration += 1  # Increment iteration
            N = self.V[2:, 1:-1]  # All the grid squares if traveling to the North.
            E = self.V[1:-1, 2:]  # All the grid squares if traveling to the East.
            S = self.V[:-2, 1:-1]  # All the grid squares if traveling to the South.
            W = self.V[1:-1, :-2]  # All the grid squares if traveling to the West.
            self.tesseract[0] = self.movement_cost + 0.8 * N + 0.1 * W + 0.1 * E  # movement cost facing North
            self.tesseract[1] = self.movement_cost + 0.8 * E + 0.1 * N + 0.1 * S  # movement cost facing East
            self.tesseract[2] = self.movement_cost + 0.8 * S + 0.1 * E + 0.1 * W  # movement cost facing South
            self.tesseract[3] = self.movement_cost + 0.8 * W + 0.1 * S + 0.1 * N  # movement cost facing West
            self.tesseract[:, self.inner != 0] = self.inner[self.inner != 0]  # Remove the fuzz from making invalid move
            self.V_1 = deepcopy(self.V)  # Update previous map
            self.V[1:-1, 1:-1] = np.max(self.tesseract, 0)  # Set current map to highest (best) direction of travel.

            # Plotting
            if LivePlotting:
                self.live_plotter(live_update=True)

    def live_plotter(self, live_update=False, Path=np.zeros([2, 1])):
        if live_update:
            self.Algorithm_policy_MDP()  # Update the current policy
            self.calc_u_v()  # Calculate the rise and run for each arrow.

        # Plotter
        fig3 = plt.figure(3)
        fig3.clf()

        # Print color map based on cost
        image3 = plt.imshow(self.V, origin='lower')
        # fig3.colorbar(image3)  # Include scale for reference
        self.plot_arrows()  # Call the plot arrows function
        plt.plot(Path[0], Path[1], linewidth=2)

        plt.draw()
        plt.pause(0.00001)

    def Algorithm_policy_MDP(self):
        self.policy = np.argmax(self.tesseract, 0)  # Set the current policy to reflect which direction is best.

    def calc_u_v(self):
        # Run for each arrow is based off of the cardinal direction
        self.u[self.policy == 0] = 0  # North
        self.u[self.policy == 1] = 1  # East
        self.u[self.policy == 2] = 0  # South
        self.u[self.policy == 3] = -1  # West

        # Rise for each arrow is based off of the cardinal direction
        self.v[self.policy == 0] = 1  # North
        self.v[self.policy == 1] = 0  # East
        self.v[self.policy == 2] = -1  # South
        self.v[self.policy == 3] = 0  # West

    def plot_arrows(self):
        # Plot arrows at x,y with slop u,v, but not on any invalid location
        plt.quiver(self.X, self.Y, self.u * self.map_inv, self.v * self.map_inv)

    def plot_path(self, x=0, y=0, LivePlotting=True):

        # Initialize lists for each path and the policy
        x_path = [x]
        y_path = [y]
        self.calc_u_v()

        while not self.goal_x_low < x <= self.goal_x_high or not self.goal_y_low < y <= self.goal_y_high:
            x_p1 = x
            y_p1 = y
            x_p1 += int(self.u[y-1, x-1])
            y_p1 += int(self.v[y-1, x-1])
            x_path.append(x_p1)
            y_path.append(y_p1)
            x = x_p1
            y = y_p1
            self.path = np.array([x_path, y_path])
            if LivePlotting:
                self.live_plotter(Path=self.path)
