#!/usr/bin/env python3

import numpy as np
from itertools import permutations as perms
import matplotlib.pyplot as plt
from copy import deepcopy
import random

Time_Horizon = 20

cost_turn_around = -1

sense_true = 0.7
sense_false = 1 - sense_true

turn_true = 0.8
turn_false = 1 - turn_true

cost_lava_f = -100
cost_lava_b = -50

cost_door_f = 100
cost_door_b = 100

meas_p1 = np.diag([sense_true, sense_false])
meas_p2 = np.diag([sense_false, sense_true])
predict_mat = np.array([[turn_false, turn_true],
                        [turn_true, turn_false]])

num_parts = 100000
step_size = 1 / num_parts

prob = np.ones([2, num_parts + 1])
prob[0] = np.arange(0, 1 + step_size, step_size)
prob[1] = np.arange(1, 0 - step_size, -step_size)
eq = np.zeros([1, 2])
eq_u1_u2 = np.array([[cost_lava_f, cost_door_b],
                     [cost_door_f, cost_lava_b]])

max_val = np.zeros(num_parts + 1)


def pruner(eq_given):
    res = eq_given @ prob
    max_val = np.amax(res, axis=0)
    indices = np.argmax(res, axis=0)
    eqs_used = np.unique(indices)
    eq_use = eq_given[eqs_used]
    return eq_use, max_val, indices


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
    eq_temp += cost_turn_around
    eq_ready = np.array([*eq_u1_u2, *eq_temp])
    return eq_ready


def plotter(max_value):
    fig = plt.figure(1)
    fig.clf()
    plt.plot(prob[0], max_value)
    plt.draw()
    plt.pause(0.001)


for T in range(Time_Horizon):
    # With Pruning
    eq_sense, max_val, indices = pruner(sensing(eq))
    eq, max_val, indices = pruner(prediction(eq_sense))
    # plotter(max_val)

    # # Without Pruning
    # eq_sense = sensing(eq)
    # eq = prediction(eq_sense)
print(eq, eq.shape)
plotter(max_val)

#### Part 5

b_p1 = 0.6
facing_lava = True
Stuck = True


def observe(B_p1, Facing_lava):
    z = deepcopy(Facing_lava)
    if random.uniform(0, 1) > sense_true:
        z = not z
    if z:
        B_p1 = sense_true * B_p1 / ((sense_true - sense_false) * B_p1 + sense_false)
    else:
        B_p1 = sense_false * B_p1 / (sense_true - (sense_true - sense_false) * B_p1)
    return B_p1


def action(B_p1, Facing_lava, stuck):
    spot = np.where(prob[0] >= B_p1)[0][0]
    act = indices[spot]
    if act == 0:
        if Facing_lava:
            print("LAVA face!!!!")
            stuck = False
        else:
            print("FREEDOM!!!!!")
            stuck = False
    elif act == 1:
        if Facing_lava:
            print("FREEDOM!!!!!")
            stuck = False
        else:
            print("I'm Lava again!!!")
            stuck = False
    else:
        print("Turning.")
        B_p1 = turn_true - (turn_true - turn_false) * B_p1
        if random.uniform(0, 1) < turn_true:
            Facing_lava = not Facing_lava
    return B_p1, Facing_lava, stuck


while Stuck:
    b_p1 = observe(b_p1, facing_lava)
    b_p1, facing_lava, Stuck = action(b_p1, facing_lava, Stuck)
