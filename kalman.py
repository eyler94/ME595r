import numpy as np

class kf:
    def __init__(self, est, P, u, meas, A, B, C, Q, R):
        self.est = est
        self.P = P
        self.u = u
        self.meas = meas
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.I = np.eye(A.shape[0])

    def update(self, est, P, u, meas):
        est_bar = self.A @ self.est + self.B @ u
        P_bar = self.A @ self.P @ self.A.T + self.Q

        K = P_bar @ self.C.T @ np.linalg.inv(self.C @ P_bar @ self.C.T + self.R)
        self.est = est_bar + K @ (meas - self.C @ est_bar)
        self.P = (self.I - K @ self.C) @ P_bar
        return self.est, self.P
