#!/usr/bin/env python3

import numpy as np
from scipy import linalg
from copy import deepcopy
from itertools import permutations as perms
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

meas_p1 = np.diag([0.7, 0.3, 0])

meas_p2 = np.diag([0.3, 0.7, 0])

predict_mat = linalg.block_diag(np.array([[0.2, 0.8], [0.8, 0.2]]), 0)

num_parts = 10000
step_size = 1 / num_parts

prob_orig = np.ones([3, num_parts + 1])
prob_orig[0] = np.arange(0, 1 + step_size, step_size)
prob_orig[1] = np.arange(1, 0 - step_size, -step_size)
#
# prob_special = deepcopy(prob_orig)
# prob_special[0] = prob_special[0] * -0.6 + 0.8
# prob_special[1] = prob_special[0] * 0.6 + 0.2

DEBUG = False


def printer(string):
    if DEBUG:
        print(string)


def maxi(eq_given):  # , special=False):
    printer("maximizing")
    prob = deepcopy(prob_orig)
    # if special:  # for special type of max
    #     prob = deepcopy(prob_special)
    res = eq_given @ prob
    max_val = np.amax(res, axis=0)
    eqs_used = np.unique(np.argmax(res, axis=0))
    eq_use = eq_given[eqs_used]
    return eq_use


def sensing(eq_given):
    printer("sensing")
    num_max = len(eq_given)
    # print("nm", num_max)
    res1 = eq_given @ meas_p1
    res2 = eq_given @ meas_p2
    pad = [(spot, spot) for spot in range(num_max)]
    indices = np.array(list(perms(range(num_max), 2)) + pad).T
    res_tot = res1[indices[0]] + res2[indices[1]]
    return res_tot


def prediction(eq_given):
    printer("Prediction.")
    eq_temp = eq_given @ predict_mat
    eq_temp[:, :-1] -= 1
    eq_ready = np.array([*eq_u1_u2, *eq_temp])
    return eq_ready


eq_u1_u2 = np.array([[-100, 100, 0],
                     [100, -50, 0]])

eq = np.array([[-100, 100, 0],
               [100, -50, 0],
               [0, 0, -1]])

for T in range(20):
    # # With Pruning
    eq_max = maxi(eq)
    eq_sense = maxi(sensing(eq_max))
    eq = maxi(prediction(eq_sense))
    # Without Pruning
    # eq_max = eq
    # eq_sense = sensing(eq_max)
    # eq = prediction(eq_sense)
    print(T)
print("eq", eq)
