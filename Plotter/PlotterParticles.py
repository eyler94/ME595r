#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

pi = np.pi

class Plotter:
    def __init__(self, x=-5, y=-3, theta=pi/2, width=20, height=20, lm=np.array([[0], [0]]), particles=np.array([[0], [0]])):
        # Properties of the world
        self.lm = lm
        self.width = width
        self.height = height

        # Properties of the robot
        self.x_loc = x
        self.y_loc = y
        self.theta = theta
        self.particles = particles

        # Points to draw the robot
        circle = np.arange(0., 2 * pi, 0.01)
        self.x_points = np.cos(circle)
        self.x_points = np.hstack([0., self.x_points])
        self.y_points = np.sin(circle)
        self.y_points = np.hstack([0., self.y_points])
        self.points = np.array([self.x_points, self.y_points])
        self.hmt = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), self.x_loc],
             [np.sin(self.theta), np.cos(self.theta), self.y_loc],
             [0., 0., 1.]])
        self.points = self.hmt @ np.asarray([self.x_points, self.y_points, np.ones([1, self.x_points.size])])

        # First plot
        fig1 = plt.figure(1)
        fig1.clf()
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(self.particles[0], self.particles[1], '.', color='black')
        plt.plot(self.points[0].T, self.points[1].T)
        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
        plt.draw()
        plt.pause(0.001)

        fig2 = plt.figure(2)
        fig2.clf()
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(self.particles[0], self.particles[1], '.', color='black')
        plt.plot(self.points[0].T, self.points[1].T)
        plt.axis([self.x_loc-3, self.x_loc+3, self.y_loc-3, self.y_loc+3])
        plt.draw()
        plt.pause(0.001)

    def update(self, x, y, theta, particles):
        self.particles = particles
        self.x_loc = x
        self.y_loc = y
        self.theta = theta
        self.calc_points()

        # Re-plot
        fig1 = plt.figure(1)
        fig1.clf()
        # Plot landmarks
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(self.particles[0], self.particles[1], '.', color='black')
        # Plot robot
        plt.plot(self.points[0].T, self.points[1].T)
        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
        plt.draw()
        plt.pause(0.001)

        # Re-plot
        fig2 = plt.figure(2)
        fig2.clf()
        # Plot landmarks
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(self.particles[0], self.particles[1], '.', color='black')
        # Plot robot
        plt.plot(self.points[0].T, self.points[1].T)
        spacing = 3
        plt.axis([self.x_loc-spacing, self.x_loc+spacing, self.y_loc-spacing, self.y_loc+spacing])
        # plt.axis([self.particles[0][0]-spacing, self.particles[0][0]+spacing, self.particles[1][0]-spacing, self.particles[1][0]+spacing])
        plt.draw()
        plt.pause(0.001)

    def calc_points(self):
        self.hmt_2d()
        self.points = self.hmt @ np.asarray([self.x_points, self.y_points, np.ones([1, self.x_points.size])])

    def hmt_2d(self):
        self.hmt = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), self.x_loc],
             [np.sin(self.theta), np.cos(self.theta), self.y_loc],
             [0., 0., 1.]])

# # Example of how to plot with
# for iter in np.arange(0., 50., 0.75):
#     Xr = np.cos(2*np.pi*iter/50.)
#     Yr = np.sin(2*np.pi*iter/50.)
#     Theta = np.pi/2.+2*np.pi*iter/50.
#     plot.update(Xr, Yr, Theta)
