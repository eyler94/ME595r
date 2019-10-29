#!/usr/bin/env python3

import numpy as np


def wrapper(ang):
    ang -= np.pi * 2 * np.floor((ang + np.pi) / (2 * np.pi))
    return ang


def inverse_range_sensor_model(square, state, meas):
    # Inverse Sensor Range Model Parameters
    alpha = 1  # wall thickness in meters
    beta = 5  # degrees
    z_max = 150  # meters

    # Grid location
    x_i = square[0]  # x location of the ith grid square
    y_i = square[1]  # y location of the ith grid square

    # Robot location
    x = state[0]  # current x location of the robot
    y = state[1]  # current y location of the robot
    theta = state[2]  # current heading of the robot

    # Measurements
    rngs = meas[0, :]  # Range measurements taken at current time
    rads = meas[1, :]  # Angles of measurements taken at current time

    r = np.sqrt((x_i - x) ** 2 + (y_i - y) ** 2)  # Range between robot and grid square
    phi = wrapper(np.arctan2(y_i - y, x_i - x) - theta)  # Angle between robot and grid square
    k = np.argmin(np.abs(phi - rads))  # k represents the measurement that is closest to that grid square
    z_k = rngs[k]  # Range measurement that is closest to the ith grid square
    theta_k = rads[k]  # Angle measurement that is closest to the ith grid square

    l_0 = 0.5
    l_occ = 1
    l_free = 0

    if r > np.min(z_max, z_k + alpha / 2) or np.abs(phi - theta_k) > beta / 2:
        return l_0
    if z_k < z_max and np.abs(r - z_k) < alpha / 2:
        return l_occ
    if r <= z_k:
        return l_free

