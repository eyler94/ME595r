import numpy as np


class Kf:
    def __init__(self, mu, SIG, u, z, A, B, C, Q, R):
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
        self.K = 0.

    def update(self, mu, SIG, u, z):
        mu_bar = self.A @ self.mu + self.B @ u
        SIG_bar = self.A @ self.SIG @ self.A.T + self.Q

        self.K = SIG_bar @ self.C.T @ np.linalg.inv(self.C @ SIG_bar @ self.C.T + self.R)
        self.mu = mu_bar + self.K @ (z - self.C @ mu_bar)
        self.SIG = (self.I - self.K @ self.C) @ SIG_bar
        return self.mu, self.SIG

    def k_return(self):
        return self.K
