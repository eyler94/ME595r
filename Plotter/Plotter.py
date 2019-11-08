#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

pi = np.pi


class Plotter:
    def __init__(self, x=-5, y=-3, theta=pi / 2, width=20, height=20, lm=np.array([[0], [0]])):
        # Properties of the world
        self.lm = lm
        self.width = width
        self.height = height

        # Properties of the robot
        self.x_loc = x
        self.y_loc = y
        self.theta = theta

        # Points to draw the robot
        circle = np.arange(0., 2 * pi, 0.01)
        self.x_points_cir = np.cos(circle)
        self.x_points = np.hstack([0., self.x_points_cir])
        self.y_points_cir = np.sin(circle)
        self.y_points = np.hstack([0., self.y_points_cir])
        self.points = np.array([self.x_points, self.y_points])
        self.cir_points = np.vstack([self.x_points_cir, self.y_points_cir])
        self.hmt = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), self.x_loc],
             [np.sin(self.theta), np.cos(self.theta), self.y_loc],
             [0., 0., 1.]])
        self.points = self.hmt @ np.asarray([self.x_points, self.y_points, np.ones([1, self.x_points.size])])

        # First plot
        fgn = plt.figure(1)
        fgn.clf()
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(self.points[0].T, self.points[1].T)
        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
        plt.draw()
        plt.pause(0.001)

    def update(self, x, y, theta):
        self.x_loc = x
        self.y_loc = y
        self.theta = theta
        self.calc_points()

        # Re-plot
        fgn = plt.figure(1)
        fgn.clf()

        # Plot landmarks
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')

        # # Plot truth and measurements
        # plt.plot(self.lm[0][0],)

        # Plot robot
        plt.plot(self.points[0].T, self.points[1].T)

        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
        plt.draw()
        plt.pause(0.001)

    def update_with_path(self, x, y, theta, true_x, true_y, mu_x, mu_y):
        self.x_loc = x
        self.y_loc = y
        self.theta = theta
        self.calc_points()

        # Re-plot
        fgn = plt.figure(1)
        fgn.clf()

        # Plot landmarks
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')

        # Plot robot
        plt.plot(self.points[0].T, self.points[1].T)

        # Plot path and estimate
        plt.plot(true_x.T, true_y.T, mu_x.T, mu_y.T)

        # plt.legend(['landmarks','robot','true path','estimated path'])
        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
        plt.draw()
        plt.pause(0.001)

    def update_with_path_and_lm(self, x, y, theta, true_x, true_y, mu_x, mu_y, mu_lm, sig_lm):
        self.x_loc = x
        self.y_loc = y
        self.theta = theta
        self.calc_points()

        # Re-plot
        fgn = plt.figure(1)
        fgn.clf()

        # Plot landmarks and estimates
        plt.plot(self.lm[0], self.lm[1], 'x', color='black')
        plt.plot(mu_lm[::2], mu_lm[1::2], 'o', color='brown')
        for spot in range(0, self.lm.shape[1]):
            u, s, v = np.linalg.svd(sig_lm[spot:spot + 2, spot:spot + 2])
            c = u @ np.diag(2*np.sqrt(s))
            ellipse = c @ self.cir_points
            plt.plot(ellipse[0]+mu_lm[spot*2], ellipse[1]+mu_lm[spot*2+1],color='green')

        # Plot robot
        plt.plot(self.points[0].T, self.points[1].T)

        # Plot path and estimate
        plt.plot(true_x.T, true_y.T, mu_x.T, mu_y.T)

        # plt.legend(['landmarks','robot','true path','estimated path'])
        plt.axis([-self.width / 2., self.width / 2., -self.height / 2., self.height / 2.])
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
