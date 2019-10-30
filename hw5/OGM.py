#!/usr/bin/env python3

import numpy as np


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


def p_to_l(p):
    return np.log(p / (1 - p))


def l_to_p(l):
    return 1 - 1 / (1 + np.exp(l))


class OGM:
    def __init__(self):
        # Inverse Sensor Range Model Parameters
        self.alpha = 1  # wall thickness in meters
        self.beta = 2 * np.pi / 180  # degrees
        self.z_max = 150  # meters
        self.p_0 = 0.5
        self.p_occ = 0.7
        self.p_free = 0.3
        self.l_0 = p_to_l(self.p_0)
        self.l_occ = p_to_l(self.p_occ)
        self.l_free = p_to_l(self.p_free)
        spots = np.arange(102).reshape([1, 102])
        one = np.ones([102, 102])
        self.x = one * spots
        # self.y = one * np.flip(spots.T)
        self.y = one * spots.T
        self.info = one * self.p_0

    def update(self, state, meas):

        # Robot location
        x_state = state[0]  # current x location of the robot
        y_state = state[1]  # current y location of the robot
        theta = state[2]  # current heading of the robot

        # Measurements
        meas = np.nan_to_num(meas, np.inf)
        rngs = meas[0, :]  # Range measurements taken at current time
        rads = meas[1, :]  # Angles of measurements taken at current time

        # Calculations
        r = np.sqrt((self.x - x_state) ** 2 + (self.y - y_state) ** 2)  # Range between robot and map
        phi = wrapper(np.arctan2(self.y - y_state, self.x - x_state) - theta)  # Angle between robot and grid square

        # Match grid squares to their appropriate measurement
        phi_diff = np.abs(phi[:, :, None] - rads[None, None, :])
        k = np.argmin(wrapper(phi_diff), 2)  # k represents the measurement that is closest to that grid square
        z_k = rngs[k]  # Range measurement that is closest to the ith grid square
        theta_k = rads[k]  # Angle measurement that is closest to the ith grid square

        # Evaluate which probabilities should be assigned to each grid square
        statement1 = np.logical_or(r > np.minimum(self.z_max, z_k + self.alpha * 0.5),
                                   np.abs(phi - theta_k) > self.beta * 0.5).astype(np.double)
        statement2 = np.logical_and(np.logical_and(z_k < self.z_max, np.abs(r - z_k) < self.alpha * 0.5),
                                    np.logical_not(statement1)).astype(np.double)
        statement3 = np.logical_and(np.logical_and(r <= z_k, np.logical_not(statement1)), np.logical_not(statement2)).astype(np.double)

        # l_total = statement1 + statement2 + statement3
        # if np.max(l_total) > 1 or np.min(l_total) < 1:
        #     print("ahhhhhhh")

        # Calculate grid probabilities
        statement1 *= p_to_l(self.info) + self.l_0 - self.l_0
        statement2 *= p_to_l(self.info) + self.l_occ - self.l_0
        statement3 *= p_to_l(self.info) + self.l_free - self.l_0

        l_total = statement1 + statement2 + statement3

        self.info = l_to_p(l_total)

    def update_itemized(self, square, state, meas):
        # Grid location
        x_i = square[0]  # x location of the ith grid square
        y_i = square[1]  # y location of the ith grid square

        # Robot location
        x_state = state[0]  # current x location of the robot
        y_state = state[1]  # current y location of the robot
        theta = state[2]  # current heading of the robot

        # Measurements
        rngs = meas[0, :]  # Range measurements taken at current time
        rads = meas[1, :]  # Angles of measurements taken at current time

        r = np.sqrt((x_i - x_state) ** 2 + (y_i - y_state) ** 2)  # Range between robot and map
        phi = wrapper(np.arctan2(y_i - y_state, x_i - x_state) - theta)  # Angle between robot and grid square
        k = np.argmin(np.abs(phi - rads))  # k represents the measurement that is closest to that grid square
        z_k = rngs[k]  # Range measurement that is closest to the ith grid square
        theta_k = rads[k]  # Angle measurement that is closest to the ith grid square

        if r > np.min(self.z_max, z_k + self.alpha / 2) or np.abs(phi - theta_k) > self.beta / 2:
            return self.l_0
        if z_k < self.z_max and np.abs(r - z_k) < self.alpha / 2:
            return self.l_occ
        if r <= z_k:
            return self.l_free
