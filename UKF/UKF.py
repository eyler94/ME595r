import numpy as np
import math
from scipy.linalg import block_diag
from scipy.linalg import cholesky as ch

def wrapper(ang):
    ang -= np.pi*2 * np.floor((ang + np.pi) / (2*np.pi))
    return ang


class UKF:
    def __init__(self, R2D2, World):
        # Generate augmented mean and covariance
        v = 1
        omega = 1
        theta = 1
        self.ts = R2D2.ts
        self.alpha1 = R2D2.alpha1
        self.alpha2 = R2D2.alpha2
        self.alpha3 = R2D2.alpha3
        self.alpha4 = R2D2.alpha4
        self.M = np.array([[self.alpha1 * v ** 2 + self.alpha2 * omega ** 2, 0],
                           [0, self.alpha3 * v ** 2 + self.alpha4 * omega ** 2]])
        self.Q = np.array([[R2D2.sigma_r**2, 0],
                           [0, R2D2.sigma_theta**2]])
        self.mu = np.array([[R2D2.x0],
                            [R2D2.y0],
                            [R2D2.theta0]])
        self.mu_a = np.array([R2D2.x0, R2D2.y0, R2D2.theta0, 0, 0, 0, 0])
        self.mu_a = np.reshape(self.mu_a,[7,1])
        self.SIG = np.diag([1, 1, 0.1])
        self.SIG_a = block_diag(self.SIG, self.M, self.Q)

        # Generate Sigma points
        self.alpha = 0.4
        self.kappa = 4
        self.beta = 2
        self.n = 7
        self.lamb_duh = self.alpha**2*(self.n+self.kappa)-self.n
        self.gamma = np.sqrt(self.n+self.lamb_duh+self.n)
        # Cholesky looks like the following L = ch(mat, lower=True)
        self.Chi_a = np.hstack([self.mu_a, self.mu_a+self.gamma*ch(self.SIG_a), self.mu_a-self.gamma*ch(self.SIG_a)])
        self.Chi_x = self.Chi_a[0:3,:]
        self.Chi_u = self.Chi_a[3:5, :]
        self.Chi_z = self.Chi_a[5:, :]

        # Pass sigma points through motion model and compute Gaussian statistics
        self.u = np.array([[0],
                           [0]])
        u = np.array([[v],
                      [omega]])
        self.Chi_bar = self.g(u+self.Chi_u,self.Chi_x)

        # Calculate weights
        self.w_m = np.ones([1,15])
        self.w_c = np.ones([1, 15])
        self.w_m[0] = self.lamb_duh/(self.n + self.lamb_duh)
        self.w_c[0] = self.w_m[0] + (1 - self.alpha**2 + self.beta)
        for spot in range(1,15):
            self.w_m[0][spot] = 1 / (2 * (self.n + self.lamb_duh))
            self.w_c[0][spot] = 1 / (2 * (self.n + self.lamb_duh))

        self.mu_bar = self.Chi_x @ self.w_m.T
        self.SIG_bar = np.multiply(self.w_c,(self.Chi_bar-self.mu_bar)) @ (self.Chi_bar-self.mu_bar).T

        # Predict observations at sigma points and compute Gaussian statistics
        self.Z_bar = self.h(self.Chi_bar) + self.Chi_z
        self.z_hat = self.Z_bar @ self.w_m.T
        self.S = np.multiply(self.w_c,(self.Z_bar-self.z_hat)) @ (self.Z_bar-self.z_hat).T
        self.SIG_xz = np.multiply(self.w_c,(self.Chi_bar-self.mu_bar)) @ (self.Z_bar-self.z_hat).T

        # Update mean and covariance
        self.K = self.SIG_xz @ np.linalg.inv(self.S)
        # # z_diff = np.array([r[spot] - z_hat[0],
        # #                    wrapper(ph[spot] - z_hat[1])])
        # z_diff =
        self.mu = self.mu_bar + self.K @ z_diff
        self.SIG = self.SIG_bar - self.K @ self.S @ self.K.T

    def g(self,u,state):
        print("Propagate sigma points.")
        v = u[0]
        omega = u[1]

        x = state[0]
        y = state[1]
        theta = state[2]

        x = x - v/omega*np.sin(theta) + v/omega*np.sin(wrapper(theta + omega * self.ts))
        y = y + v/omega*np.cos(theta) - v/omega*np.cos(wrapper(theta + omega * self.ts))
        theta = wrapper(theta + omega * self.ts)

        return x, y, theta


    def h(self,x):
        print("Collect sigma measurements.")



    def update(self, mu, SIG, v, omega, r, ph):
        self.mu = mu
        self.SIG = SIG
        theta = self.mu[2][0]
        th_om_dt = wrapper(theta + omega * self.ts)
        self.G = np.array([[1, 0, - v / omega * np.cos(theta) + v / omega * np.cos(th_om_dt)],
                           [0, 1, - v / omega * np.sin(theta) + v / omega * np.sin(th_om_dt)],
                           [0, 0, 1]])
        self.V = np.array([[(-np.sin(theta)+np.sin(th_om_dt))/omega, v*(np.sin(theta)-np.sin(th_om_dt))/omega**2 + v*np.cos(th_om_dt)*self.ts/omega],
                           [(np.cos(theta)-np.cos(th_om_dt))/omega, -v*(np.cos(theta)-np.cos(th_om_dt))/omega**2 + v*np.sin(th_om_dt)*self.ts/omega],
                           [0., self.ts]])
        self.M = np.array([[self.alpha1 * v ** 2 + self.alpha2 * omega ** 2, 0],
                           [0, self.alpha3 * v ** 2 + self.alpha4 * omega ** 2]])
        self.mu_bar = self.mu + np.array([[- v / omega * np.sin(theta) + v / omega * np.sin(th_om_dt)],
                                          [v / omega * np.cos(theta) - v / omega * np.cos(th_om_dt)],
                                          [omega*self.ts]])
        self.SIG_bar = self.G @ self.SIG @ self.G.T + self.V @ self.M @ self.V.T
        self.features(r, ph)

        self.mu = self.mu_bar
        self.SIG = self.SIG_bar

        return self.mu, self.SIG

    def features(self, r, ph):
        for spot in range(0, self.num_landmarks):
            diff_x = self.landmarks[0][spot]-self.mu_bar[0][0]
            diff_y = self.landmarks[1][spot]-self.mu_bar[1][0]
            q = diff_x ** 2 + diff_y ** 2
            z_hat = np.array([[np.sqrt(q)],
                              [wrapper(math.atan2(diff_y, diff_x)-self.mu_bar[2][0])]])
            H = np.array([[-diff_x/np.sqrt(q), -diff_y/np.sqrt(q), 0],
                          [diff_y/q, -diff_x/q, -1]])
            S = H @ self.SIG_bar @ H.T + self.Q
            K = self.SIG_bar @ H.T @ np.linalg.inv(S)
            z_diff = np.array([r[spot]-z_hat[0],
                               wrapper(ph[spot]-z_hat[1])])
            self.mu_bar = self.mu_bar + K @ z_diff
            self.SIG_bar = (np.eye(3) - K @ H) @ self.SIG_bar




