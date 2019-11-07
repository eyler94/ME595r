import numpy as np
from scipy.linalg import block_diag


def wrapper(ang):
    if ang > np.pi:
        # print("Too much.")
        ang = ang - 2 * np.pi
    elif ang <= -np.pi:
        # print("Too little.")
        ang = ang + 2 * np.pi
    return ang


class ekf_slam:
    def __init__(self, R2D2, World):
        print("Init")
        self.F = np.hstack([np.eye(3), np.zeros([3, 2 * World.Number_Landmarks])])
        v = 1
        omega = 1
        theta = 1
        self.ts = R2D2.ts
        th_om_dt = wrapper(theta + omega * self.ts)
        self.mu_state = np.array([[R2D2.x0],
                                  [R2D2.y0],
                                  [R2D2.theta0]])
        self.mu_landmarks = np.zeros([World.Number_Landmarks*2, 1])
        self.mu = np.vstack([self.mu_state, self.mu_landmarks])
        self.mu_bar = self.calc_g(v, theta, omega, th_om_dt)
        self.G = self.calc_G(theta, th_om_dt, v, omega)
        self.V = self.calc_V(theta, th_om_dt, v, omega)
        self.alpha1 = R2D2.alpha1
        self.alpha2 = R2D2.alpha2
        self.alpha3 = R2D2.alpha3
        self.alpha4 = R2D2.alpha4
        self.M = np.array([[self.alpha1 * v ** 2 + self.alpha2 * omega ** 2, 0],
                           [0, self.alpha3 * v ** 2 + self.alpha4 * omega ** 2]])
        state_block = np.diag([1, 1, 0.1])
        lm_block = np.zeros([World.Number_Landmarks*2, World.Number_Landmarks*2])
        self.SIG = block_diag(state_block, lm_block)
        self.SIG_bar = self.G @ self.SIG @ self.G.T + self.F.T @ self.V @ self.M @ self.V.T @ self.F
        self.Q = np.array([[R2D2.sigma_r ** 2, 0],
                           [0, R2D2.sigma_theta ** 2]])
        self.landmarks = World.Landmarks
        self.num_landmarks = World.Number_Landmarks

    def calc_g(self, v, theta, omega, th_om_dt):
        mat = np.array([[- v / omega * np.sin(theta) + v / omega * np.sin(th_om_dt)],
                        [v / omega * np.cos(theta) - v / omega * np.cos(th_om_dt)],
                        [omega * self.ts]])
        mu_bar = self.mu + self.F.T @ mat
        return mu_bar

    def calc_G(self, theta, th_om_dt, v, omega):
        G = np.eye(9) + self.F.T @ np.array([[0, 0, - v / omega * np.cos(theta) + v / omega * np.cos(th_om_dt)],
                                             [0, 0, - v / omega * np.sin(theta) + v / omega * np.sin(th_om_dt)],
                                             [0, 0, 0]]) @ self.F
        return G

    def calc_V(self, theta, th_om_dt, v, omega):
        V = np.array([[(-np.sin(theta) + np.sin(th_om_dt)) / omega,
                       v * (np.sin(theta) - np.sin(th_om_dt)) / omega ** 2 + v * np.cos(
                           th_om_dt) * self.ts / omega],
                      [(np.cos(theta) - np.cos(th_om_dt)) / omega,
                       -v * (np.cos(theta) - np.cos(th_om_dt)) / omega ** 2 + v * np.sin(
                           th_om_dt) * self.ts / omega],
                      [0., self.ts]])
        return V

    def update(self, mu, SIG, v, omega, r, ph):
        self.mu = mu
        self.SIG = SIG
        theta = self.mu[2][0]
        th_om_dt = wrapper(theta + omega * self.ts)
        self.G = self.calc_G(theta, th_om_dt, v, omega)
        self.V = self.calc_V(theta, th_om_dt, v, omega)
        self.M = np.array([[self.alpha1 * v ** 2 + self.alpha2 * omega ** 2, 0],
                           [0, self.alpha3 * v ** 2 + self.alpha4 * omega ** 2]])
        self.mu_bar = self.calc_g(v, theta, omega, th_om_dt)
        self.SIG_bar = self.G @ self.SIG @ self.G.T + self.V @ self.M @ self.V.T
        self.features(r, ph)

        self.mu = self.mu_bar
        self.SIG = self.SIG_bar

        return self.mu, self.SIG

    def features(self, r, ph):
        for spot in range(0, self.num_landmarks):
            diff_x = self.landmarks[0][spot] - self.mu_bar[0][0]
            diff_y = self.landmarks[1][spot] - self.mu_bar[1][0]
            q = diff_x ** 2 + diff_y ** 2
            z_hat = np.array([[np.sqrt(q)],
                              [wrapper(np.arctan2(diff_y, diff_x) - self.mu_bar[2][0])]])
            H = np.array([[-diff_x / np.sqrt(q), -diff_y / np.sqrt(q), 0],
                          [diff_y / q, -diff_x / q, -1]])
            S = H @ self.SIG_bar @ H.T + self.Q
            K = self.SIG_bar @ H.T @ np.linalg.inv(S)
            z_diff = np.array([r[spot] - z_hat[0],
                               wrapper(ph[spot] - z_hat[1])])
            self.mu_bar = self.mu_bar + K @ z_diff
            self.SIG_bar = (np.eye(3) - K @ H) @ self.SIG_bar
