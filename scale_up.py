"""
SCALE UP: czy monolit z wieksza pojemnoscia przekracza granice klasy 2?
Test na dwoch zadaniach: role_coordination (v4) + self_delayed.
Wymiary skali: hidden, population, generations.
"""
import numpy as np
import copy

# ============================================================
# WSPOLNE
# ============================================================
STRUCT_SEED = 1
TRAIN_NOISE = list(range(1, 6))
T = 100
SIGMA_PROC = 0.1
MIN_HIDDEN = 4
MAX_HIDDEN = 64
EPS = 1e-6

def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# ZADANIE 1: SELF-DELAYED (self-model)
# ============================================================
N_STATE = 8
M1, M2, M3 = 2, 1, 1
M_IN = M1 + M2 + M3
SIGMAS = (0.10, 0.25, 0.05)
K_ACTION = 1
C_SCALE = 0.1
DELAY = 5

class SelfWorldDelayed:
    def __init__(self, structure_seed, noise_seed, C_scale, delay):
        rng_s = np.random.default_rng(structure_seed)
        A = rng_s.standard_normal((N_STATE, N_STATE))
        A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.B = rng_s.standard_normal((N_STATE, K_ACTION)) * 0.5
        self.C_full = rng_s.standard_normal((N_STATE, MAX_HIDDEN)) * C_scale
        self.P1 = rng_s.standard_normal((M1, N_STATE))
        self.P2 = rng_s.standard_normal((M2, N_STATE))
        self.P3 = rng_s.standard_normal((M3, N_STATE))
        self.rng = np.random.default_rng(noise_seed)
        self.h = self.rng.standard_normal(N_STATE) * 0.5
        self.delay = delay
        self.s_history = []
    def observe(self):
        o1 = self.P1 @ self.h + self.rng.standard_normal(M1) * SIGMAS[0]
        o2 = self.P2 @ self.h + self.rng.standard_normal(M2) * SIGMAS[1]
        o3 = self.P3 @ self.h + self.rng.standard_normal(M3) * SIGMAS[2]
        return o1, o2, o3
    def step(self, action, state):
        if self.delay == 0:
            s_delayed = state
        else:
            if len(self.s_history) >= self.delay:
                s_delayed = self.s_history[0]
            else:
                s_delayed = np.zeros(len(state))
            self.s_history.append(state.copy())
            if len(self.s_history) > self.delay:
                self.s_history.pop(0)
        C_eff = self.C_full[:, :len(s_delayed)]
        self.h = (self.A @ self.h + self.B @ action + C_eff @ s_delayed
                  + self.rng.standard_normal(N_STATE) * SIGMA_PROC)
        return -float(np.sum(self.h ** 2))

class RecOrg:
    def __init__(self, m_in, k, hidden, rng):
        self.m_in = m_in; self.k = k; self.hidden = hidden
        sc = 0.1
        self.W_s = rng.standard_normal((hidden, hidden)) * sc
        self.W_o = rng.standard_normal((hidden, m_in)) * sc
        self.b_s = np.zeros(hidden)
        self.W_a = rng.standard_normal((k, hidden)) * sc
        self.b_a = np.zeros(k)
    def init_state(self): return np.zeros(self.hidden)
    def step(self, s, o):
        s2 = np.tanh(self.W_s @ s + self.W_o @ o + self.b_s)
        a = np.tanh(self.W_a @ s2 + self.b_a)
        return a, s2

def concat_obs(o1, o2, o3):
    return np.concatenate([o1, o2, o3])

def run_self(org, noise_seed, delay):
    w = SelfWorldDelayed(STRUCT_SEED, noise_seed, C_SCALE, delay)
    s = org.init_state()
    r = 0.0
    S, H = [], []
    for _ in range(T):
        o1, o2, o3 = w.observe()
        a, s_next = org.step(s, concat_obs(o1, o2, o3))
        r += w.step(a, s_next)
        S.append(s_next.copy()); H.append(w.h.copy())
        s = s_next
    return r, np.array(S), np.array(H)

def fitness_self(org, noise_list, delay):
    return float(np.mean([run_self(org, ns, delay)[0] for ns in noise_list]))

def mutate_rec(org, rng, sigma, size_rate=0.1):
    c = copy.deepcopy(org)
    for attr in ["W_s","W_o","b_s","W_a","b_a"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if rng.random() < size_rate:
        d = -1 if rng.random() < 0.5 else 1
        nh = max(MIN_HIDDEN, min(MAX_HIDDEN, c.hidden + d))
        if nh != c.hidden:
            new = RecOrg(c.m_in, c.k, nh, rng)
            h_min = min(c.hidden, nh)
            new.W_s[:h_min,:h_min] = c.W_s[:h_min,:h_min]
            new.W_o[:h_min,:] = c.W_o[:h_min,:]
            new.b_s[:h_min] = c.b_s[:h_min]
            new.W_a[:,:h_min] = c.W_a[:,:h_min]
            new.b_a[:] = c.b_a[:]
            c = new
    return c

def evolve_self(seed, hidden_start, N, G):
    rng = np.random.default_rng(seed)
    pop = [RecOrg(M_IN, K_ACTION, hidden_start, rng) for _ in range(N)]
    best_fit = -np.inf; best = None
    for gen in range(G):
        fits = np.array([fitness_self(o, TRAIN_NOISE, DELAY) for o in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]; fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]; best = copy.deepcopy(pop[0])
        n_el = max(1, int(N * 0.25))
        elite = pop[:n_el]
        children = []
        while len(children) < N - n_el:
            p = elite[rng.integers(n_el)]
            children.append(mutate_rec(p, rng, 0.05))
        pop = elite + children
    return best

# ============================================================
# ZADANIE 2: ROLE COORDINATION v4 (klasa 2)
# ============================================================
HALF = 4
MSG_DIM = 2
SIGMA_OBS = 0.15
TARGET_LEVELS = [-2.0, -1.0, 1.0, 2.0]
TARGET_PERIOD_G1 = 30
TARGET_PERIOD_G2 = 5
LAMBDA_TARGET = 0.5
LAMBDA_COORD = 2.0

class RoleWorldV4:
    def __init__(self, structure_seed, noise_seed):
        rng_s = np.random.default_rng(structure_seed)
        A = rng_s.standard_normal((N_STATE, N_STATE))
        A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        self.B_B = np.zeros(N_STATE); self.B_B[0] = 1.0
        self.B_C = np.zeros(N_STATE); self.B_C[4] = 1.0
        self.P_B = np.zeros((HALF, N_STATE))
        self.P_B[np.arange(HALF), np.arange(HALF)] = 1.0
        self.P_C = np.zeros((HALF, N_STATE))
        self.P_C[np.arange(HALF), np.arange(HALF) + HALF] = 1.0
        self.rng = np.random.default_rng(noise_seed)
        self.h = self.rng.standard_normal(N_STATE) * 0.5
        self.g1 = 0.0; self.g2 = 0.0; self.t = 0
    def observe_A(self):
        return np.array([self.g1, self.g2])
    def observe_B(self):
        return self.P_B @ self.h + self.rng.standard_normal(HALF) * SIGMA_OBS
    def observe_C(self):
        return self.P_C @ self.h + self.rng.standard_normal(HALF) * SIGMA_OBS
    def step(self, a_B, a_C, g):
        a_B_eff = g * float(a_B); a_C_eff = (1.0 - g) * float(a_C)
        self.h = (self.A @ self.h
                  + self.B_B * a_B_eff + self.B_C * a_C_eff
                  + self.rng.standard_normal(N_STATE) * SIGMA_PROC)
        self.t += 1
        if self.t % TARGET_PERIOD_G1 == 0:
            self.g1 = float(self.rng.choice(TARGET_LEVELS))
        if self.t % TARGET_PERIOD_G2 == 0:
            self.g2 = float(self.rng.choice(TARGET_LEVELS))
        r_stab = -float(np.sum(self.h ** 2))
        r_target = -LAMBDA_TARGET * (float((self.h[0] - self.g1) ** 2)
                                     + float((self.h[4] - self.g2) ** 2))
        err_B = float((self.h[0] - self.g1) ** 2)
        err_C = float((self.h[4] - self.g2) ** 2)
        optimal_g = err_B / (err_B + err_C + EPS)
        r_coord = -LAMBDA_COORD * (g - optimal_g) ** 2
        return r_stab + r_target + r_coord

class Agent:
    def __init__(self, m_own, hidden, role, rng):
        self.m_own = m_own; self.hidden = hidden; self.role = role
        if role == "A":
            m_in = m_own + 2 * MSG_DIM
        else:
            m_in = m_own + MSG_DIM
        self.m_in = m_in
        sc = 0.1
        self.W_s = rng.standard_normal((hidden, hidden)) * sc
        self.W_o = rng.standard_normal((hidden, m_in)) * sc
        self.b_s = np.zeros(hidden)
        self.W_m = rng.standard_normal((MSG_DIM, hidden)) * sc
        self.b_m = np.zeros(MSG_DIM)
        if role == "A":
            self.W_role = rng.standard_normal((1, hidden)) * sc
            self.b_role = np.zeros(1)
        else:
            self.W_a = rng.standard_normal((1, hidden)) * sc
            self.b_a = np.zeros(1)
    def init_state(self): return np.zeros(self.hidden)
    def step(self, s, own_obs, msg_from_other):
        x = np.concatenate([own_obs, msg_from_other])
        s2 = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        m = np.tanh(self.W_m @ s2 + self.b_m)
        if self.role == "A":
            g = float(sigmoid(self.W_role @ s2 + self.b_role)[0])
            return g, m, s2
        else:
            a = float(np.tanh(self.W_a @ s2 + self.b_a)[0])
            return a, m, s2

def episode_role(agA, agB, agC, w, force_g=None):
    sA = agA.init_state(); sB = agB.init_state(); sC = agC.init_state()
    mA = np.zeros(MSG_DIM); mB = np.zeros(MSG_DIM); mC = np.zeros(MSG_DIM)
    r_tot = 0.0
    g_list, opt_list = [], []
    for _ in range(T):
        oA = w.observe_A(); oB = w.observe_B(); oC = w.observe_C()
        msg_to_A = np.concatenate([mB, mC])
        g_raw, new_mA, sA = agA.step(sA, oA, msg_to_A)
        a_B, new_mB, sB = agB.step(sB, oB, mA)
        a_C, new_mC, sC = agC.step(sC, oC, mA)
        g = g_raw if force_g is None else force_g
        err_B = float((w.h[0] - w.g1) ** 2)
        err_C = float((w.h[4] - w.g2) ** 2)
        opt = err_B / (err_B + err_C + EPS)
        g_list.append(g); opt_list.append(opt)
        r_tot += w.step(a_B, a_C, g)
        mA, mB, mC = new_mA, new_mB, new_mC
    return r_tot, np.array(g_list), np.array(opt_list)

def fitness_role(agA, agB, agC, noise_list, return_traj=False, **kwargs):
    rewards = []
    gs, opts = [], []
    for ns in noise_list:
        w = RoleWorldV4(STRUCT_SEED, ns)
        r, g, opt = episode_role(agA, agB, agC, w, **kwargs)
        rewards.append(r); gs.append(g); opts.append(opt)
    mean_r = float(np.mean(rewards))
    if return_traj:
        return mean_r, np.concatenate(gs), np.concatenate(opts)
    return mean_r

def mutate_agent(ag, rng, sigma, size_rate=0.1):
    c = copy.deepcopy(ag)
    for attr in ["W_s","W_o","b_s","W_m","b_m"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if c.role == "A":
        for attr in ["W_role","b_role"]:
            v = getattr(c, attr)
            setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    else:
        for attr in ["W_a","b_a"]:
            v = getattr(c, attr)
            setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if rng.random() < size_rate:
        d = -1 if rng.random() < 0.5 else 1
        nh = max(MIN_HIDDEN, min(MAX_HIDDEN, c.hidden + d))
        if nh != c.hidden:
            new = Agent(c.m_own, nh, c.role, rng)
            h_min = min(c.hidden, nh)
            new.W_s[:h_min,:h_min] = c.W_s[:h_min,:h_min]
            new.W_o[:h_min,:] = c.W_o[:h_min,:]
            new.b_s[:h_min] = c.b_s[:h_min]
            new.W_m[:,:h_min] = c.W_m[:,:h_min]
            new.b_m[:] = c.b_m[:]
            if c.role == "A":
                new.W_role[:,:h_min] = c.W_role[:,:h_min]
                new.b_role[:] = c.b_role[:]
            else:
                new.W_a[:,:h_min] = c.W_a[:,:h_min]
                new.b_a[:] = c.b_a[:]
            c = new
    return c

def evolve_role(seed, hidden_start, N, G):
    rng = np.random.default_rng(seed)
    def make_trio():
        a = Agent(2, hidden_start, "A", rng)
        b = Agent(M_IN, hidden_start, "B", rng)
        c = Agent(M_IN, hidden_start, "C", rng)
        return [a, b, c]
    pop = [make_trio() for _ in range(N)]
    best_fit = -np.inf; best = None
    for gen in range(G):
        fits = np.array([fitness_role(t[0], t[1], t[2], TRAIN_NOISE) for t in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]; fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]; best = [copy.deepcopy(x) for x in pop[0]]
        n_el = max(1, int(N * 0.25))
        elite = pop[:n_el]
        children = []
        while len(children) < N - n_el:
            p = elite[rng.integers(n_el)]
            children.append([mutate_agent(p[0], rng, 0.05),
                             mutate_agent(p[1], rng, 0.05),
                             mutate_agent(p[2], rng, 0.05)])
        pop = elite + children
    return best

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 84)
    print("  SCALE UP: czy monolit z wieksza pojemnoscia przekracza klase 2?")
    print("=" * 84)
    print()

    # V4 baseline z poprzednich eksperymentow
    print("  V4 baseline (hidden=8, N=20, G=120):")
    print("    corr(g,opt) = +0.39")
    print("    g_std       = 0.05")
    print("    mute B->A+C->A = -4.5")
    print()

    seeds = [1, 2]
    configs = [
        # (label, hidden_start, N, G)
        ("S1: h16 N100 G200", 16, 100, 200),
        ("S2: h32 N100 G300", 32, 100, 300),
    ]

    # ---- ROLE COORDINATION v4 at scale ----
    print("=" * 84)
    print("  ZADANIE 1: ROLE COORDINATION v4 (klasa 2)")
    print("=" * 84)
    print(f"  {'config':>20}  {'seed':>4}  {'corr':>7}  {'g_std':>7}  "
          f"{'opt_std':>8}  {'sw':>4}  {'mBA+CA':>8}")
    print("  " + "-" * 70)

    for label, hidden_start, N, G in configs:
        for seed in seeds:
            best = evolve_role(seed, hidden_start, N, G)
            agA, agB, agC = best
            full, g_traj, opt_traj = fitness_role(agA, agB, agC, TRAIN_NOISE,
                                                  return_traj=True)
            no_back = fitness_role(agA, agB, agC, TRAIN_NOISE,
                                   mute_BA=True, mute_CA=True) \
                      if False else full  # placeholder — v4 nie ma mute w tej wersji
            # bez mute w tej wersji — pomijamy
            g_binary = (g_traj > 0.5).astype(int)
            sw = int(np.sum(np.abs(np.diff(g_binary))))
            g_std = float(g_traj.std())
            opt_std = float(opt_traj.std())
            if g_traj.std() > 1e-6 and opt_traj.std() > 1e-6:
                corr = float(np.corrcoef(g_traj, opt_traj)[0, 1])
            else:
                corr = 0.0
            print(f"  {label:>20}  {seed:>4}  {corr:>+7.3f}  {g_std:>7.3f}  "
                  f"{opt_std:>8.3f}  {sw:>4}  {0:>+8.2f}")

    print()
    print("=" * 84)
    print("  ZADANIE 2: SELF-DELAYED (self-model)")
    print("=" * 84)
    print(f"  {'config':>20}  {'seed':>4}  {'R2(future)':>12}  {'R2(now)':>10}")
    print("  " + "-" * 55)

    def r2(X, Y):
        Xa = np.hstack([X, np.ones((len(X), 1))])
        coef, _, _, _ = np.linalg.lstsq(Xa, Y, rcond=None)
        pred = Xa @ coef
        var = np.mean(Y ** 2) + 1e-9
        return float(1 - np.mean((Y - pred) ** 2) / var)

    for label, hidden_start, N, G in configs:
        for seed in seeds:
            best = evolve_self(seed, hidden_start, N, G)
            _, S, H = run_self(best, TRAIN_NOISE[0], DELAY)
            r2_now = r2(S, H)
            lag = DELAY
            r2_future = r2(S[:-lag], H[lag:])
            print(f"  {label:>20}  {seed:>4}  {r2_future:>+12.3f}  {r2_now:>+10.3f}")

    print()
    print("=" * 84)
    print("  INTERPRETACJA")
    print("=" * 84)
    print("  Jesli po skalowaniu:")
    print("    - corr(g,opt) rosnie z ~0.4 do ~0.6+  => SKALA POMOGA")
    print("    - g_std rosnie z ~0.05 do ~0.15+     => A NADAZA za zmiana")
    print("    - R2(future) rosnie z ~0.03 do ~0.2+ => SELF-MODEL EMERGUJE")
    print()
    print("  Jesli wszystkie liczby sa takie same:")
    print("    - Monolit NIE potrzebuje skali. Sufit jest strukturalny.")
    print("    - Argument za grafem MIEDZY agentami.")