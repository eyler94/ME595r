#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

pi = np.pi

class Plotter:
    def __init__(self, x=-5, y=-3, theta=pi/2, width=20, height=20, lm=np.array([[0], [0]])):
        self.x = x
        self.y = y
        self.theta = theta
        self.lm = lm
        self.width = width
        self.height = height
        circle = np.arange(0., 2 * np.pi, 0.01)
        x = np.cos(circle)
        x = np.hstack([0., x])
        y = np.sin(circle)
        y = np.hstack([0., y])
        self.points = np.array([x, y])
        self.fig = plt.figure(1)
        self.hmt = np.array([[np.cos(self.theta), -np.sin(self.theta), self.x], [np.sin(self.theta), np.cos(self.theta), self.y], [0., 0., 1.]])


    def update(self, x, y, theta):
        print("Update")

