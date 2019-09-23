import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')

# Instantiate World, Robot, and EKF

# Set Timeline
Tf = 20     # Sec
Ts = 0.1    # Sec
time_data = np.arange(0., Tf, Ts)

# Loop through Sim

