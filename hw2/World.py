#!/usr/bin/env python3

# Class defining a world for the autonomous robot (R2D2) to wander around. It has a collection of landmarks
# and a plotting function.

import numpy as np

Usually_width = 20
Usually_height = 20


class World:
    def __init__(self, width=Usually_width, height=Usually_height):
        # Size parameters
        self.width = width
        self.height = height

        # Landmarks
        self.Number_Landmarks = 3
        self.Landmarks = np.array([[6, -7, 6], [4, 8, -4]])


class Landmark:
    def __init__(self, x=np.random.randint(-Usually_width / 2., Usually_width / 2.),
                 y=np.random.randint(-Usually_height / 2., Usually_height / 2.)):
        self.x = x
        self.y = y
