"""
COUPLED_TRACKING 5 AGENTS 1BIT: test skalowania z waskim kanalem.
A + B,C,D,E. A widzi 1D + 4 binarne wiadomosci.
Cel: przewidziec z(t+1) w 2D z waskim kanalem.
"""
import numpy as np
import copy

# ============================================================
# PARAMETRY
# ============================================================
N_STATE = 2
PERIOD = 8
DECAY = 0.95
SIGMA_Z = 0.05
SIGMA_OBS = 0.1
T = 100
HORIZON = 1
HIDDEN = 16
N_POP = 30
G = 100
SEEDS = list(range(1, 11))
TRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(100, 106))

N_ENCODERS = 4
SATURATION = 5.0

# ============================================================
# SWIAT
# ============================================================
class OscWorld2D:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        theta = 2 * np.pi / PERIOD
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        self.A = R * DECAY
        self.rng = rng
        self.z = rng.standard_normal(N_STATE) * 0.5
        self.P_A = rng.standard_normal((1, N_STATE))
        self.P_A /= np.linalg.norm(self.P_A)
        self.P_enc = []
        for _ in range(N_ENCODERS):
            p = rng.standard_normal((1, N_STATE))
            p /= np.linalg.norm(p)
            self.P_enc.append(p)
        self.t = 0
    def observe_A(self):
        return self.P_A @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def observe_enc(self, i):
        return self.P_enc[i] @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def step(self):
        self.z = self.A @ self.z + self.rng.standard_normal(N_STATE) * SIGMA_Z
        self.t += 1
    def get_z(self):
        return self.z.copy()

# ============================================================
# AGENT
# ============================================================
class IntegratorA:
    def __init__(self, m_own, n_channels, rng):
        self.m_own = m_own
        self.n_channels = n_channels
        m_in = m_own + n_channels
        self.m_in = m_in
        sc = 0.1
        self.W_s = rng.standard_normal((HIDDEN, HIDDEN)) * sc
        self.W_o = rng.standard_normal((HIDDEN, m_in)) * sc
        self.b_s = np.zeros(HIDDEN)
        self.W_pred = rng.standard_normal((N_STATE, HIDDEN)) * sc
        self.b_pred = np.zeros(N_STATE)
        self.channels = np.ones(n_channels)
    def init_state(self):
        return np.zeros(HIDDEN)
    def step(self, s, obs, msgs):
        masked = np.array([self.channels[i] * msgs[i] for i in range(self.n_channels)])
        x = np.concatenate([obs, masked])
        s_new = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        z_pred = self.W_pred @ s_new + self.b_pred
        return z_pred, s_new

class EncoderAgent:
    def __init__(self, m_own, rng):
        self.m_own = m_own
        sc = 0.1
        self.W_s = rng.standard_normal((HIDDEN, HIDDEN)) * sc
        self.W_o = rng.standard_normal((HIDDEN, m_own)) * sc
        self.b_s = np.zeros(HIDDEN)
        self.W_msg = rng.standard_normal((1, HIDDEN)) * sc
        self.b_msg = np.zeros(1)
    def init_state(self):
        return np.zeros(HIDDEN)
    def step(self, s, obs):
        s_new = np.tanh(self.W_s @ s + self.W_o @ obs + self.b_s)
        raw = float((self.W_msg @ s_new + self.b_msg)[0])
        msg = float(np.tanh(SATURATION * raw))
        return msg, s_new

# ============================================================
# EPIZOD
# ============================================================
def episode(agA, encoders, w, mute_mask=None, record=False):
    sA = agA.init_state()
    s_enc = [e.init_state() for e in encoders]
    prev_msgs = [0.0] * N_ENCODERS
    rA_tot = 0.0
    z_preds = []
    z_trues = []
    if mute_mask is None:
        mute_mask = np.ones(N_ENCODERS)
    for _ in range(T):
        oA = w.observe_A()
        obs_enc = [w.observe_enc(i) for i in range(N_ENCODERS)]
        msgs_in = [prev_msgs[i] * mute_mask[i] for i in range(N_ENCODERS)]
        z_pred_A, sA = agA.step(sA, oA, msgs_in)
        new_msgs = []
        for i, e in enumerate(encoders):
            msg, s_enc[i] = e.step(s_enc[i], obs_enc[i])
            new_msgs.append(msg)
        w.step()
        z_next = w.get_z()
        rA_tot -= float(np.sum((z_pred_A - z_next) ** 2))
        z_preds.append(z_pred_A.copy())
        z_trues.append(z_next.copy())
        prev_msgs = new_msgs
    if record:
        return rA_tot, np.array(z_preds), np.array(z_trues)
    return rA_tot, None, None

def evaluate(agA, encoders, seeds, mute_mask=None):
    total = 0.0
    for s in seeds:
        w = OscWorld2D(s)
        r, _, _ = episode(agA, encoders, w, mute_mask=mute_mask)
        total += r
    return total / len(seeds)

# ============================================================
# MUTACJA
# ============================================================
def mutate_A(ag, rng, sigma):
    c = copy.deepcopy(ag)
    for attr in ["W_s", "W_o", "b_s", "W_pred", "b_pred"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    for i in range(c.n_channels):
        if rng.random() < 0.05:
            c.channels[i] = 1.0 - c.channels[i]
    return c

def mutate_enc(ag, rng, sigma):
    c = copy.deepcopy(ag)
    for attr in ["W_s", "W_o", "b_s", "W_msg", "b_msg"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    return c

# ============================================================
# EWOLUCJA
# ============================================================
def evolve(seed, N=N_POP, G=G):
    rng = np.random.default_rng(seed)
    def make_pop():
        a = IntegratorA(1, N_ENCODERS, rng)
        encs = [EncoderAgent(1, rng) for _ in range(N_ENCODERS)]
        return [a, encs]
    pop = [make_pop() for _ in range(N)]
    best_fit = -np.inf
    best = None
    for gen in range(G):
        fits = np.array([evaluate(t[0], t[1], TRAIN_SEEDS) for t in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]
        fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]
            best = [copy.deepcopy(pop[0][0]),
                    [copy.deepcopy(e) for e in pop[0][1]]]
        n_el = max(1, int(N * 0.25))
        elite = pop[:n_el]
        children = []
        while len(children) < N - n_el:
            p = elite[rng.integers(n_el)]
            new_a = mutate_A(p[0], rng, 0.05)
            new_encs = [mutate_enc(e, rng, 0.05) for e in p[1]]
            children.append([new_a, new_encs])
        pop = elite + children
    return best, best_fit

# ============================================================
# METRYKI
# ============================================================
def r2_raw(pred, actual):
    mse = np.mean(np.sum((pred - actual) ** 2, axis=1))
    var = np.mean(np.sum((actual - actual.mean(axis=0)) ** 2, axis=1))
    return float(1.0 - mse / (var + 1e-12))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 84)
    print("  COUPLED_TRACKING 5 AGENTS 1BIT")
    print("=" * 84)
    print(f"  z in R^2, kanal binarny (tanh({SATURATION}*x))")
    print(f"  A widzi 1D + {N_ENCODERS} wiadomosci")
    print(f"  Test: czy waski kanal przelamuje granice network size?")
    print()

    results = []
    for seed in SEEDS:
        best, bf = evolve(seed, G=G)
        agA, encs = best
        w = OscWorld2D(TEST_SEEDS[0])
        _, z_pred_full, z_true = episode(agA, encs, w, record=True)
        r2_full = r2_raw(z_pred_full, z_true)
        deltas = []
        for i in range(N_ENCODERS):
            mask = np.ones(N_ENCODERS)
            mask[i] = 0.0
            w = OscWorld2D(TEST_SEEDS[0])
            _, z_pred_m, z_true_m = episode(agA, encs, w, mute_mask=mask, record=True)
            r2_m = r2_raw(z_pred_m, z_true_m)
            deltas.append(r2_full - r2_m)
        mask_all = np.zeros(N_ENCODERS)
        w = OscWorld2D(TEST_SEEDS[0])
        _, z_pred_all, z_true_all = episode(agA, encs, w, mute_mask=mask_all, record=True)
        r2_all_mute = r2_raw(z_pred_all, z_true_all)
        dBC = r2_full - r2_all_mute
        ch = agA.channels
        active_ch = int(ch.sum())
        results.append((seed, r2_full, deltas, ch.copy(), active_ch, dBC))
        print(f"  seed={seed:>2}: full={r2_full:+.3f}  ch_active={active_ch}/{N_ENCODERS}  "
              f"dBC={dBC:+.3f}  deltas=[{', '.join(f'{d:+.2f}' for d in deltas)}]")

    print()
    print("=" * 84)
    print("  PODSUMOWANIE 10 SEEDOW")
    print("=" * 84)

    deltas_arr = np.array([r[2] for r in results])
    mean_delta = deltas_arr.mean(axis=0)
    n_used = np.array([(deltas_arr[:, i] > 0.05).sum() for i in range(N_ENCODERS)])
    full_arr = np.array([r[1] for r in results])
    dBC_arr = np.array([r[5] for r in results])
    active_arr = np.array([r[4] for r in results])

    print(f"  full R2: mean={full_arr.mean():+.3f}  min={full_arr.min():+.3f}  max={full_arr.max():+.3f}")
    print(f"  dBC:     mean={dBC_arr.mean():+.3f}  min={dBC_arr.min():+.3f}  max={dBC_arr.max():+.3f}")
    print()
    print(f"  Sredni wklad kanalu:")
    for i in range(N_ENCODERS):
        print(f"    kanal {i+1}: delta mean={mean_delta[i]:+.3f}  "
              f"uzywany w {n_used[i]}/10 seedow")
    print()
    print(f"  Aktywne kanaly: mean={active_arr.mean():.2f}/4")
    print()
    print("=" * 84)
    print("  POROWNANIE")
    print("=" * 84)
    print(f"  3-agent 2D ciagly:   full R2 = +0.404, dBC = +0.442")
    print(f"  3-agent 3D binarny:  full R2 = +0.123, dBC = +0.568")
    print(f"  5-agent 2D binarny:  full R2 = {full_arr.mean():+.3f}, dBC = {dBC_arr.mean():+.3f}, "
          f"dBC>0.05 w {int((dBC_arr > 0.05).sum())}/10")
    print()
    print("=" * 84)
    print("  WERDYKT")
    print("=" * 84)
    if full_arr.mean() > 0.2 and (dBC_arr > 0.05).sum() >= 7:
        print(f"  >>> WASKI KANAL PRZELAMAL GRANICE NETWORK SIZE.")
        print(f"      full R2 = {full_arr.mean():+.3f}, dBC>0.05 w {int((dBC_arr > 0.05).sum())}/10.")
        print(f"      Regula skalowania potwierdzona.")
    elif full_arr.mean() > -0.1:
        print(f"  >>> CZESCIOWA POPRAWA. full R2 = {full_arr.mean():+.3f}.")
        print(f"      Waski kanal pomaga, ale network size nadal ogranicza.")
    else:
        print(f"  >>> WASKI KANAL NIE PRZELAMUJE NETWORK SIZE.")
        print(f"      full R2 = {full_arr.mean():+.3f}. 5 agentow to twarda granica.")