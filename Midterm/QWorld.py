#!/usr/bin/env python3

# Class defining a world for the midterm quadrotor. It has a collection of landmarks

import numpy as np

Usually_width = 30
Usually_height = 30


class World:
    def __init__(self, width=Usually_width, height=Usually_height):
        # Size parameters
        self.width = width
        self.height = height

        # Landmarks
        rand = False
        if rand:
            self.Number_Landmarks = 50
            self.Landmarks = np.random.randint(-10, 10, [2, self.Number_Landmarks])
        else:
            self.Number_Landmarks = 6
            self.Landmarks = np.array([[6, -7, 12, -2, -10, 13], [4, 8, -8, 0, 2, 7]])
