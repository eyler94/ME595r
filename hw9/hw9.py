#!/usr/bin/env python3

import numpy as np
from itertools import permutations as perms

meas_p1 = np.diag([0.7, 0.3])
meas_p2 = np.diag([0.3, 0.7])
predict_mat = np.array([[0.2, 0.8], [0.8, 0.2]])

num_parts = 100000
step_size = 1 / num_parts

prob = np.ones([2, num_parts + 1])
prob[0] = np.arange(0, 1 + step_size, step_size)
prob[1] = np.arange(1, 0 - step_size, -step_size)
eq = np.zeros([1, 2])
eq_u1_u2 = np.array([[-100, 100],
                     [100, -50]])


def pruner(eq_given):
    res = eq_given @ prob
    max_val = np.amax(res, axis=0)
    eqs_used = np.unique(np.argmax(res, axis=0))
    eq_use = eq_given[eqs_used]
    return eq_use


def sensing(eq_given):
    num_max = len(eq_given)
    result_1 = eq_given @ meas_p1
    result_2 = eq_given @ meas_p2
    pad = [(spot, spot) for spot in range(num_max)]
    indices = np.array(list(perms(range(num_max), 2)) + pad).T
    res_tot = result_1[indices[0]] + result_2[indices[1]]
    return res_tot


def prediction(eq_given):
    eq_temp = eq_given @ predict_mat
    eq_temp -= 1
    eq_ready = np.array([*eq_u1_u2, *eq_temp])
    return eq_ready


for T in range(20):
    # With Pruning
    eq_max = pruner(eq)
    eq_sense = pruner(sensing(eq_max))
    eq = pruner(prediction(eq_sense))

    # # Without Pruning
    # eq_sense = sensing(eq)
    # eq = prediction(eq_sense)
print(eq, eq.shape)
