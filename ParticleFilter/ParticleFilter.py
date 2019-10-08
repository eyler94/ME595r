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


class ParticalFilter:
    def __init__(self, mu, SIG):
        print("Initializing Particle Filter.")
        self.mu = mu
        self.SIG = SIG

    def update(self, mu, SIG):
        print("Updating Particle Filter.")
        self.mu = mu
        self.SIG = SIG