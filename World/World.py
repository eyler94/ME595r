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
        rand = True
        if rand:
            self.Number_Landmarks = 50
            self.Landmarks = np.random.randint(-10, 10, [2, self.Number_Landmarks])
        else:
            self.Number_Landmarks = 3
            self.Landmarks = np.array([[6, -7, 6], [4, 8, -4]])


