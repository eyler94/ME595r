import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import inv
from scipy.linalg import det


def wrapper(ang):
    if ang > np.pi:
        # print("Too much.")
        ang = ang - 2 * np.pi
    elif ang <= -np.pi:
        # print("Too little.")
        ang = ang + 2 * np.pi
    return ang


class Particle:
    def __init__(self, num_landmarks):
        self.state = np.zeros([3, 1])
        self.SIG = np.eye(3) * 100.
        self.landmarks = [Landmark() for ii in range(num_landmarks)]


class Landmark:
    def __init__(self):
        self.mu = np.zeros([2, 1])
        self.SIG = np.eye(3) * 1e5


class FastSlam:
    def __init__(self, R2D2, World):
        print("Init")
        # World Properties
        self.landmarks = World.Landmarks
        self.num_landmarks = World.Number_Landmarks
        self.alpha1 = R2D2.alpha1
        self.alpha2 = R2D2.alpha2
        self.alpha3 = R2D2.alpha3
        self.alpha4 = R2D2.alpha4
        self.alpha5 = R2D2.alpha5
        self.alpha6 = R2D2.alpha6
        self.u = np.array([[R2D2.v_c],
                           [R2D2.omega_c]])
        self.z = R2D2.calculate_measurements(self.num_landmarks, self.landmarks)
        self.ts = R2D2.ts
        self.num_states = 3
        self.sigma_r = R2D2.sigma_r
        self.sigma_theta = R2D2.sigma_theta

        self.Q_t = np.array([[R2D2.sigma_r ** 2, 0],
                             [0, R2D2.sigma_theta ** 2]])

        self.M = 10
        self.M_inv = 1 / self.M
        self.particles = [Particle(self.num_landmarks) for ii in range(self.M)]

        self.p0 = 1 / self.M

        self.weights = np.ones([self.M, 1])*self.p0
        self.mu = np.zeros([3, 1])
        self.mu_lm = np.zeros([self.num_landmarks, 2])
        self.SIG = np.zeros([3, 3])
        self.SIG_lm = np.eye(self.num_landmarks, self.num_landmarks)*1e5

    def prediction(self, particle, v, omega):
        v_hat = v + np.random.randn(1) * np.sqrt(self.alpha1 * v ** 2 + self.alpha2 * omega ** 2)
        omega_hat = omega + np.random.randn(1) * np.sqrt(self.alpha3 * v ** 2 + self.alpha4 * omega ** 2)
        gamma_hat = np.random.randn(1) * np.sqrt(self.alpha5 * v ** 2 + self.alpha6 * omega ** 2)

        theta = particle.state[2]

        particle.state[0] += - v_hat / omega_hat * np.sin(theta) + v_hat / omega_hat * np.sin(
            wrapper(theta + omega_hat * self.ts))
        particle.state[1] += + v_hat / omega_hat * np.cos(theta) - v_hat / omega_hat * np.cos(
            wrapper(theta + omega_hat * self.ts))
        particle.state[2] = wrapper(theta + omega_hat * self.ts + gamma_hat * self.ts)

    def features(self, particle, r, ph, particle_spot):
        for landmark_spot in range(0, self.num_landmarks):
            if not np.isnan(r[landmark_spot]):  # if the measurement does not return a nan (meaning it was seen)
                if particle.landmarks[landmark_spot].mu[0][
                    0] == 0:  # if the x of the landmark == 0.0, it hasn't been seen
                    particle.landmarks[landmark_spot].mu[0][0] = particle.state[0][0] + r[landmark_spot] * np.cos(
                        ph[landmark_spot] + particle.state[2][0])  # x of lm
                    particle.landmarks[landmark_spot].mu[1][0] = particle.state[1][0] + r[landmark_spot] * np.cos(
                        ph[landmark_spot] + particle.state[2][0])  # y of lm
                    dx = particle.landmarks[landmark_spot].mu[0][0] - particle.state[0][0]
                    dy = particle.landmarks[landmark_spot].mu[1][0] - particle.state[1][0]
                    q = dx ** 2 + dy ** 2
                    H = 1 / q * np.array([[np.sqrt(q) * dx, np.sqrt(q) * dy],
                                          [-dy, dx]])
                    particle.landmarks[landmark_spot].SIG = inv(H) @ self.Q_t @ inv(H).T
                    self.weights[particle_spot][0] = self.p0
                else:
                    dx = particle.landmarks[landmark_spot].mu[0][0] - particle.state[0][0]
                    dy = particle.landmarks[landmark_spot].mu[1][0] - particle.state[1][0]
                    q = dx ** 2 + dy ** 2
                    z_hat = np.array([[np.sqrt(q)],
                                      [wrapper(np.arctan2(dy, dx) - particle.state[2][0])]])
                    H = 1 / q * np.array([[np.sqrt(q) * dx, np.sqrt(q) * dy],
                                          [-dy, dx]])
                    Q = H @ particle.landmarks[landmark_spot].SIG @ H.T + self.Q_t
                    K = particle.landmarks[landmark_spot].SIG @ H.T @ inv(Q)
                    z_diff = np.array([r[landmark_spot] - z_hat[0],
                                       wrapper(ph[landmark_spot] - z_hat[1])])
                    particle.landmarks[landmark_spot].mu += K @ z_diff
                    particle.landmarks[landmark_spot].SIG = (np.eye(2) - K @ H) @ particle.landmarks[landmark_spot].SIG
                    self.weights[particle_spot][0] = 1 / np.sqrt(det(2 * np.pi * Q)) * np.exp(
                        -0.5 * z_diff.T @ inv(Q) @ z_diff)

    def low_variance_sampler(self):
        particles_bar = [Particle(self.num_landmarks) for ii in range(self.M)]
        r = np.random.rand() * self.M_inv
        c = self.weights[0]
        i = 0
        indx = np.array([])
        for m in range(self.M):
            U = r + m * self.M_inv
            while U > c:
                i = i + 1
                c = c + self.weights[i]
            particles_bar[m] = self.particles[i]
            indx = np.hstack([indx, i])

        return particles_bar

    def update(self, r, ph, v, omega, Y):
        for particle_spot in range(self.M):
            particle = self.particles[particle_spot]
            self.prediction(particle, v, omega)
            self.features(particle, r, ph, particle_spot)

        self.weights = self.weights / np.sum(self.weights)
        self.particles = self.low_variance_sampler()
