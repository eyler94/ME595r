import numpy as np
import matplotlib.pyplot as plt

size = 100

real = 74
truth = np.ones([size,1])*real

# Estimate parameters
est = 73 # estimate
est_u = 2 # estimate uncertainty
est_log = np.zeros([size,1])
est_log[0,0] = est

# Measurement parameters
meas = 75 # measurement
meas_u = 1 # measurement uncertainty
meas_log = np.zeros([size,1])
meas_log[0,0] = meas

for iter in range(1,size):
    # Calculate the kalman gain
    KG = est_u / (est_u + meas_u)

    # Calculate the current estimate
    est = est + KG * (meas-est)
    est_log[iter,0] = est

    # Calculate the uncertainty in the estimate
    est_u = (1 - KG) * est_u

    # Collect new measurement
    meas = real + np.random.randn()*meas_u
    meas_log[iter,0] = meas

    print("temperature:", est)

time = np.arange(0,size).reshape([size,1])

plt.plot(time,truth)
plt.plot(time,est_log)
plt.plot(time,meas_log)
plt.show()
print("Closest:",np.amin(np.abs(real-est_log)))
