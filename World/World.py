#!/usr/bin/env python3

# Class defining a world for the autonomous robot (R2D2) to wander around. It has a collection of landmarks

import numpy as np

Usually_width = 20
Usually_height = 20


class World:
    def __init__(self, width=Usually_width, height=Usually_height, rand=False, num_lm=5):
        # Size parameters
        self.width = width
        self.height = height

        # Landmarks
        if rand:
            self.Number_Landmarks = num_lm
            self.Landmarks = np.random.randint(-10, 10, [2, self.Number_Landmarks])
        else:
            self.Number_Landmarks = 3
            self.Landmarks = np.array([[6, -7, 6], [4, 8, -4]])
