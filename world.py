import numpy as np


class World:
    def __init__(self, n=6, m1=3, m2=2, m3=1, k=1,
                 rho=0.9, sigma_process=0.1,
                 sigmas=(0.10, 0.25, 0.05),
                 structure_seed=1, noise_seed=None,
                 action_delay=0):
        self.n, self.m1, self.m2, self.m3, self.k = n, m1, m2, m3, k
        self.sigma_process = sigma_process
        self.sigmas = sigmas
        self.action_delay = action_delay
        self.action_buffer = []

        rng_s = np.random.default_rng(structure_seed)
        A = rng_s.standard_normal((n, n))
        A *= rho / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.B = rng_s.standard_normal((n, k)) * 0.5
        self.P1 = rng_s.standard_normal((m1, n))
        self.P2 = rng_s.standard_normal((m2, n))
        self.P3 = rng_s.standard_normal((m3, n))

        ns = noise_seed if noise_seed is not None else structure_seed
        self.rng = np.random.default_rng(ns)

        self.h = self.rng.standard_normal(n) * 0.5
        self.t = 0

    def observe(self):
        o1 = self.P1 @ self.h + self.rng.standard_normal(self.m1) * self.sigmas[0]
        o2 = self.P2 @ self.h + self.rng.standard_normal(self.m2) * self.sigmas[1]
        o3 = self.P3 @ self.h + self.rng.standard_normal(self.m3) * self.sigmas[2]
        return o1, o2, o3

    def step(self, action):
        if self.action_delay <= 0:
            a_eff = action
        else:
            self.action_buffer.append(action.copy())
            if len(self.action_buffer) > self.action_delay:
                a_eff = self.action_buffer.pop(0)
            else:
                a_eff = np.zeros(self.k)

        self.h = (self.A @ self.h
                  + self.B @ a_eff
                  + self.rng.standard_normal(self.n) * self.sigma_process)
        self.t += 1
        return -float(np.sum(self.h ** 2))


if __name__ == "__main__":
    T = 200
    w = World(structure_seed=1, noise_seed=1, action_delay=0)
    r = 0.0
    for _ in range(T): r += w.step(np.array([0.5]))
    print(f"delay=0  reward(a=0.5) = {r:.3f}")

    w = World(structure_seed=1, noise_seed=1, action_delay=3)
    r = 0.0
    for _ in range(T): r += w.step(np.array([0.5]))
    print(f"delay=3  reward(a=0.5) = {r:.3f}")