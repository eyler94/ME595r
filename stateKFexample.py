import numpy as np
import matplotlib.pyplot as plt
from kalman import kf

#This is going to model a projectile

size = 250

k = 0
b = 0
g = -9.81
u = np.array([[0., g]]).T
m = 1
V0 = 50
Ts = 0.04
theta = np.radians(15)

X_1 = np.array([[0,np.cos(theta)*V0,0,np.sin(theta)*V0]]).T
X = np.array([[0.,0.,0.,0.]]).T

a = np.array([[1., Ts],[0, (1.-b)]])
A = np.zeros([4,4])
A[0:2,0:2] = a
A[2:,2:] = a

b = np.array([[(Ts**2.)/2],[Ts]])
B = np.array([[(Ts**2.)/2, 0.],[Ts, 0.],[0., (Ts**2.)/2],[0., Ts]])

print(B)

c = np.array([1., 0.])
C = np.array([[1., 0., 0., 0.],[0.,0.,1.,0.]])

truth = np.zeros([size,2])
truth[0,:]=X_1.item(0), X_1.item(2)

# Model
P = np.eye(4)*1e-2 # model uncertainty
est = X_1# Original estimate
Q = np.eye(4)*1e-2# State transition uncertainty
est_log = np.zeros([size,2])
est_log[0,:] = est.item(0),est.item(2)

# Measurements
delta = 0.5#3125
meas = C@X+np.array([[1.],[1.]])*np.random.randn()*delta # First measurement
meas_log = np.zeros([size,2])
meas_log[0,:] = meas.item(0),meas.item(1)
R = np.eye(2)*delta # Measurement uncertainty

filter = kf(est,P,u,meas,A,B,C,Q,R)

for iter in range(1,size):

    # Compute truth
    X = A@X_1 + B@u
    truth[iter,0]=X.item(0)
    X_1=X
    if X.item(2)>0.:
        truth[iter,1] = X.item(2)
        X_1 = X
    else:
        break

    # Collect measurement
    meas = C@X+np.array([[1.],[1.]])*np.random.randn()*delta
    meas_log[iter, :] = meas.item(0), meas.item(1)

    est, P = filter.update(X,P,u,meas)
    est_log[iter, :] = est.item(0), est.item(2)

plt.plot(truth[:iter,0],truth[:iter,1])
plt.axis([-1, np.amax(meas_log)+1, -1, np.amax(meas_log)/2.])
plt.plot(est_log[:iter,0],est_log[:iter,1])
plt.plot(meas_log[:iter,0],meas_log[:iter,1])
plt.show()
