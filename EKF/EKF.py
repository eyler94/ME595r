import numpy as np


class EKF:
    def __init__(self):
        self.mu = mu
        self.SIG = SIG
        self.u = u
        self.z = z
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.I = np.eye(A.shape[0])

    def update(self, mu, SIG, u, z):
        self.mu = mu
        self.SIG = SIG
        mu_bar = self.A @ self.mu + self.B * u
        SIG_bar = self.A @ self.SIG @ self.A.T + self.R

        self.K = SIG_bar @ self.C.T @ np.linalg.inv(self.C @ SIG_bar @ self.C.T + self.Q)
        self.mu = mu_bar + self.K @ (z - self.C @ mu_bar)
        self.SIG = (self.I - self.K @ self.C) @ SIG_bar
        return self.mu, self.SIG