"""
COOPERATION v2: A zna target, B nie. A musi zakodowac target w komunikacie.
B musi doprowadzic h[0] do targetu. Bez semantyki = porazka.
"""
import numpy as np
import copy

STRUCT_SEED = 1
TRAIN_NOISE = list(range(1, 6))
TEST_NOISE  = list(range(1000, 1006))
T = 100
N_STATE = 8
HALF = 4
COMM_DIM = 4           # wiekszy kanal, bo musi zmiescic target
SIGMA_PROC = 0.1
SIGMA_OBS  = 0.15
MIN_HIDDEN = 2
MAX_HIDDEN = 16
TARGET_LEVELS = [-2.0, -1.0, 1.0, 2.0]
TARGET_PERIOD = 20     # co ile krokow zmienia sie target
LAMBDA_TARGET = 0.5    # waga bledu targetu

class TargetWorld:
    def __init__(self, structure_seed, noise_seed):
        rng_s = np.random.default_rng(structure_seed)
        A = rng_s.standard_normal((N_STATE, N_STATE))
        A *= 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
        self.A = A
        # B dziala na wszystkich wymiarach
        self.B_B = np.ones(N_STATE) * 0.3
        self.P_A = np.zeros((HALF, N_STATE))
        self.P_A[np.arange(HALF), np.arange(HALF)] = 1.0
        self.P_B = np.zeros((HALF, N_STATE))
        self.P_B[np.arange(HALF), np.arange(HALF) + HALF] = 1.0
        self.rng = np.random.default_rng(noise_seed)
        self.h = self.rng.standard_normal(N_STATE) * 0.5
        # target g zmienia sie co TARGET_PERIOD krokow
        self.g = 0.0
        self.t = 0
    def get_target(self):
        return self.g
    def observe_A(self):
        # A widzi dims 0-3 + sygnal targetu
        o = self.P_A @ self.h + self.rng.standard_normal(HALF) * SIGMA_OBS
        return np.concatenate([o, [self.g]])
    def observe_B(self):
        # B widzi dims 4-7 BEZ targetu
        return self.P_B @ self.h + self.rng.standard_normal(HALF) * SIGMA_OBS
    def step(self, a_B):
        self.h = (self.A @ self.h
                  + self.B_B * float(a_B)
                  + self.rng.standard_normal(N_STATE) * SIGMA_PROC)
        self.t += 1
        # aktualizacja targetu
        if self.t % TARGET_PERIOD == 0:
            self.g = float(self.rng.choice(TARGET_LEVELS))
        # nagroda: stabilnosc + dopasowanie h[0] do targetu
        r_stab = -float(np.sum(self.h ** 2))
        r_target = -LAMBDA_TARGET * float((self.h[0] - self.g) ** 2)
        return r_stab + r_target

class CommOrg:
    def __init__(self, m_own, comm_dim, hidden, rng):
        self.m_own = m_own
        self.comm_dim = comm_dim
        self.hidden = hidden
        m_in = m_own + comm_dim
        sc = 0.1
        self.W_s = rng.standard_normal((hidden, hidden)) * sc
        self.W_o = rng.standard_normal((hidden, m_in)) * sc
        self.b_s = np.zeros(hidden)
        self.W_m = rng.standard_normal((comm_dim, hidden)) * sc  # wyjscie komunikacyjne
        self.b_m = np.zeros(comm_dim)
        self.W_a = rng.standard_normal((1, hidden)) * sc         # wyjscie akcji (tylko B)
        self.b_a = np.zeros(1)
    def init_state(self): return np.zeros(self.hidden)
    def step(self, s, o, m_in):
        x = np.concatenate([o, m_in])
        s2 = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        m_out = np.tanh(self.W_m @ s2 + self.b_m)
        a = float(np.tanh(self.W_a @ s2 + self.b_a)[0])
        return a, m_out, s2

def episode(org_A, org_B, w, T=T, mute_A=False, random_msg=False):
    s_A = org_A.init_state(); s_B = org_B.init_state()
    m_A = np.zeros(COMM_DIM)
    m_B = np.zeros(COMM_DIM)
    rng = np.random.default_rng(123)
    r_tot = 0.0
    for _ in range(T):
        oA = w.observe_A()
        oB = w.observe_B()
        # A dziala, produkuje komunikat
        a_A, new_m_A, s_A = org_A.step(s_A, oA, m_B)
        # komunikat dla B
        if mute_A:
            msg_to_B = np.zeros(COMM_DIM)
        elif random_msg:
            msg_to_B = rng.standard_normal(COMM_DIM)
        else:
            msg_to_B = new_m_A
        # B dostaje komunikat i produkuje akcje
        a_B, new_m_B, s_B = org_B.step(s_B, oB, msg_to_B)
        r_tot += w.step(a_B)
        m_A = new_m_A; m_B = new_m_B
    return r_tot

def evaluate(org_A, org_B, noise_list, **kwargs):
    return float(np.mean([episode(org_A, org_B, TargetWorld(STRUCT_SEED, ns), **kwargs)
                          for ns in noise_list]))

def collect_trajectory(org_A, org_B, noise_seed):
    w = TargetWorld(STRUCT_SEED, noise_seed)
    s_A = org_A.init_state(); s_B = org_B.init_state()
    m_A = np.zeros(COMM_DIM); m_B = np.zeros(COMM_DIM)
    traj = {"m_A": [], "g": []}
    for _ in range(T):
        oA = w.observe_A()
        oB = w.observe_B()
        a_A, new_m_A, s_A = org_A.step(s_A, oA, m_B)
        a_B, new_m_B, s_B = org_B.step(s_B, oB, new_m_A)
        traj["m_A"].append(new_m_A.copy())
        traj["g"].append(w.get_target())
        w.step(a_B)
        m_A = new_m_A; m_B = new_m_B
    return traj

def resize(org, new_hidden, rng):
    if new_hidden == org.hidden: return org
    new = CommOrg(org.m_own, org.comm_dim, new_hidden, rng)
    h_min = min(org.hidden, new_hidden)
    new.W_s[:h_min, :h_min] = org.W_s[:h_min, :h_min]
    new.W_o[:h_min, :]      = org.W_o[:h_min, :]
    new.b_s[:h_min]         = org.b_s[:h_min]
    new.W_m[:, :h_min]      = org.W_m[:, :h_min]
    new.b_m[:]              = org.b_m[:]
    new.W_a[:, :h_min]      = org.W_a[:, :h_min]
    new.b_a[:]              = org.b_a[:]
    return new

def mutate(org, rng, sigma, size_rate=0.1):
    c = copy.deepcopy(org)
    for attr in ["W_s","W_o","b_s","W_m","b_m","W_a","b_a"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if rng.random() < size_rate:
        d = -1 if rng.random() < 0.5 else 1
        nh = max(MIN_HIDDEN, min(MAX_HIDDEN, c.hidden + d))
        if nh != c.hidden: c = resize(c, nh, rng)
    return c

def evolve(seed, N=20, G=200, elite=0.25, sigma=0.05):
    rng = np.random.default_rng(seed)
    pairs = [(CommOrg(HALF + 1, COMM_DIM, int(rng.integers(MIN_HIDDEN, MAX_HIDDEN + 1)), rng),
              CommOrg(HALF, COMM_DIM, int(rng.integers(MIN_HIDDEN, MAX_HIDDEN + 1)), rng))
             for _ in range(N)]
    hist = []
    for gen in range(G):
        fits = np.array([evaluate(A, B, TRAIN_NOISE) for (A, B) in pairs])
        idx = np.argsort(fits)[::-1]
        pairs = [pairs[i] for i in idx]; fits = fits[idx]
        hA_m = np.mean([A.hidden for (A, B) in pairs])
        hB_m = np.mean([B.hidden for (A, B) in pairs])
        hist.append((gen, hA_m, hB_m, fits[0], fits.mean()))
        n_el = max(1, int(N * elite))
        elite_pairs = pairs[:n_el]
        new_pairs = list(elite_pairs)
        while len(new_pairs) < N:
            A, B = elite_pairs[rng.integers(n_el)]
            new_pairs.append((mutate(A, rng, sigma), mutate(B, rng, sigma)))
        pairs = new_pairs
    return pairs, hist

if __name__ == "__main__":
    print("=" * 80)
    print("  COOPERATION v2: A zna target, B nie. A musi go zakodowac.")
    print("=" * 80)
    print(f"  COMM_DIM={COMM_DIM}  TARGET_PERIOD={TARGET_PERIOD}  LAMBDA_TARGET={LAMBDA_TARGET}")
    print()

    # baseline: losowe
    rng0 = np.random.default_rng(0)
    A0 = CommOrg(HALF + 1, COMM_DIM, 4, rng0)
    B0 = CommOrg(HALF, COMM_DIM, 4, rng0)
    base = evaluate(A0, B0, TRAIN_NOISE)
    print(f"  baseline (random):  {base:.3f}")
    print()

    results = []
    for seed in [1, 2, 3]:
        print(f"--- seed={seed} ---")
        pairs, hist = evolve(seed, G=150)
        A, B = pairs[0]
        full = evaluate(A, B, TRAIN_NOISE)
        mute = evaluate(A, B, TRAIN_NOISE, mute_A=True)
        rnd  = evaluate(A, B, TRAIN_NOISE, random_msg=True)
        test = evaluate(A, B, TEST_NOISE)
        print(f"  {'gen':>4}  {'hA':>5}  {'hB':>5}  {'best':>9}  {'mean':>9}")
        for row in hist[::30] + [hist[-1]]:
            g, hA_m, hB_m, bf, mf = row
            print(f"  {g:>4}  {hA_m:>5.2f}  {hB_m:>5.2f}  {bf:>9.3f}  {mf:>9.3f}")
        print(f"  best: hA={A.hidden}  hB={B.hidden}")
        print(f"  full:   {full:.3f}")
        print(f"  mute:   {mute:.3f}   delta={mute - full:+.3f}")
        print(f"  random: {rnd:.3f}   delta={rnd - full:+.3f}")
        print(f"  test:   {test:.3f}")
        # semantyka: R^2(m_A -> g)
        trajs = [collect_trajectory(A, B, ns) for ns in TRAIN_NOISE]
        m_A_all = np.concatenate([t["m_A"] for t in trajs], axis=0)
        g_all = np.concatenate([t["g"] for t in trajs], axis=0)
        X = np.hstack([m_A_all, np.ones((len(m_A_all), 1))])
        coef, _, _, _ = np.linalg.lstsq(X, g_all, rcond=None)
        pred = X @ coef
        var_g = np.mean(g_all ** 2) + 1e-6
        mse = np.mean((g_all - pred) ** 2)
        r2 = 1 - mse / var_g
        print(f"  R^2(m_A -> g) = {r2:+.3f}")
        results.append((seed, A.hidden, B.hidden, full, mute, rnd, test, r2))
        print()

    print("=" * 80)
    print("  PODSUMOWANIE")
    print("=" * 80)
    print(f"  {'seed':>5}  {'hA':>4}  {'hB':>4}  {'full':>10}  {'mute':>10}  {'random':>10}  {'R^2(g)':>9}")
    print("  " + "-" * 72)
    for (s, hA, hB, f, m, r, t, r2) in results:
        print(f"  {s:>5}  {hA:>4}  {hB:>4}  {f:>10.3f}  {m:>10.3f}  {r:>10.3f}  {r2:>+9.3f}")

    print()
    print("=" * 80)
    print("  KRYTERIA SUKCESU")
    print("=" * 80)
    print("  1) mute < full - 5    => komunikacja istotna")
    print("  2) random < full - 5  => tresc komunikatu istotna")
    print("  3) R^2(m_A -> g) > 0.3 => A koduje target")
    print("  4) test ~ full        => generalizuje")
    print("  5) hA lub hB > 6      => semantyka wymaga struktury")