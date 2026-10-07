"""
COUPLED_TRACKING 5 AGENTS: A, B, C, D, E.
A: integrator, widzi 1D projekcje z + wiadomosci od B, C, D, E
B-E: widza 2D z (rozne projekcje), wysylaja 1D do A
Kanaly sa EWOLUOWALNE - kazdy moze sie wlaczyc/wylaczyc.
Test: hierarchia czy mesh? Ile kanalow przetrwa?
"""
import numpy as np
import copy

# ============================================================
# PARAMETRY
# ============================================================
PERIOD = 8
DECAY = 0.95
SIGMA_Z = 0.05
SIGMA_OBS = 0.1
T = 100
HIDDEN = 16
N_POP = 30
G = 100
SEEDS = list(range(1, 11))
TRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(100, 106))

N_ENCODERS = 4  # B, C, D, E

# ============================================================
# SWIAT
# ============================================================
class OscWorld:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        theta = 2 * np.pi / PERIOD
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        self.A = R * DECAY
        self.rng = rng
        self.z = rng.standard_normal(2) * 0.5
        # A: 1D projekcja
        self.P_A = rng.standard_normal((1, 2))
        self.P_A /= np.linalg.norm(self.P_A)
        # B-E: rozne projekcje 2D->1D (każdy widzi 1D)
        self.P_enc = []
        for _ in range(N_ENCODERS):
            p = rng.standard_normal((1, 2))
            p /= np.linalg.norm(p)
            self.P_enc.append(p)
        self.t = 0
    def observe_A(self):
        return self.P_A @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def observe_enc(self, i):
        return self.P_enc[i] @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def step(self):
        self.z = self.A @ self.z + self.rng.standard_normal(2) * SIGMA_Z
        self.t += 1
    def get_z(self):
        return self.z.copy()

# ============================================================
# AGENT
# ============================================================
class IntegratorA:
    """A: widzi 1D + N wiadomosci, ewoluowalne kanaly."""
    def __init__(self, m_own, n_channels, rng):
        self.m_own = m_own
        self.n_channels = n_channels
        m_in = m_own + n_channels
        self.m_in = m_in
        sc = 0.1
        self.W_s = rng.standard_normal((HIDDEN, HIDDEN)) * sc
        self.W_o = rng.standard_normal((HIDDEN, m_in)) * sc
        self.b_s = np.zeros(HIDDEN)
        self.W_pred = rng.standard_normal((2, HIDDEN)) * sc
        self.b_pred = np.zeros(2)
        # kanaly: 0 = wylaczony, 1 = wlaczony
        self.channels = np.ones(n_channels)
    def init_state(self):
        return np.zeros(HIDDEN)
    def step(self, s, obs, msgs):
        # msgs: lista dlugosci n_channels
        masked = np.array([self.channels[i] * msgs[i] for i in range(self.n_channels)])
        x = np.concatenate([obs, masked])
        s_new = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        z_pred = self.W_pred @ s_new + self.b_pred
        return z_pred, s_new

class EncoderAgent:
    """B-E: widzi 1D, wysyla 1D wiadomosc do A."""
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
        msg = float(np.tanh(self.W_msg @ s_new + self.b_msg)[0])
        return msg, s_new

# ============================================================
# EPIZOD
# ============================================================
def episode(agA, encoders, w, mute_mask=None, record=False):
    """mute_mask: wektor 0/1 dlugosci N_ENCODERS — ktore kanaly wyciszyc."""
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
        # apply mute
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
        w = OscWorld(s)
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
    # mutuj kanaly
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
    print(f"  COUPLED_TRACKING {N_ENCODERS + 1} AGENTS")
    print("=" * 84)
    print(f"  A: integrator (1D + {N_ENCODERS} wiadomosci, kanaly ewoluowalne)")
    print(f"  B,C,D,E: encodery (1D, wysylaja do A)")
    print()

    results = []
    for seed in SEEDS:
        best, bf = evolve(seed, G=G)
        agA, encs = best
        w = OscWorld(TEST_SEEDS[0])
        _, z_pred_full, z_true = episode(agA, encs, w, record=True)
        r2_full = r2_raw(z_pred_full, z_true)
        # ablacja: kazdy kanal osobno
        deltas = []
        for i in range(N_ENCODERS):
            mask = np.ones(N_ENCODERS)
            mask[i] = 0.0
            w = OscWorld(TEST_SEEDS[0])
            _, z_pred_m, z_true_m = episode(agA, encs, w, mute_mask=mask, record=True)
            r2_m = r2_raw(z_pred_m, z_true_m)
            deltas.append(r2_full - r2_m)
        # ablacja: wszystko
        mask_all = np.zeros(N_ENCODERS)
        w = OscWorld(TEST_SEEDS[0])
        _, z_pred_all, z_true_all = episode(agA, encs, w, mute_mask=mask_all, record=True)
        r2_all_mute = r2_raw(z_pred_all, z_true_all)
        # deltas[i] to wklad kanalu i
        ch = agA.channels
        active_ch = int(ch.sum())
        results.append((seed, r2_full, deltas, ch.copy(), active_ch))
        print(f"  seed={seed:>2}: full={r2_full:+.3f}  "
              f"ch_active={active_ch}/{N_ENCODERS}  "
              f"deltas=[{', '.join(f'{d:+.2f}' for d in deltas)}]")

    print()
    print("=" * 84)
    print("  PODSUMOWANIE 10 SEEDOW")
    print("=" * 84)

    # jak czesto kazdy kanal jest uzywany
    deltas_arr = np.array([r[2] for r in results])
    mean_delta = deltas_arr.mean(axis=0)
    n_used = np.array([(deltas_arr[:, i] > 0.05).sum() for i in range(N_ENCODERS)])
    print(f"  Sredni wklad kanalu:")
    for i in range(N_ENCODERS):
        print(f"    kanal {i+1}: delta mean={mean_delta[i]:+.3f}  "
              f"uzywany w {n_used[i]}/10 seedow")

    print()
    active_arr = np.array([r[4] for r in results])
    print(f"  Aktywne kanaly (z ewolucji):")
    print(f"    mean={active_arr.mean():.2f}/4  min={active_arr.min()}  max={active_arr.max()}")
    print()

    # klasyfikacja topologii
    print(f"  Klasyfikacja topologii:")
    for i in range(N_ENCODERS):
        if n_used[i] >= 7:
            print(f"    kanal {i+1}: STALY (>=7/10 seedow)")
        elif n_used[i] >= 4:
            print(f"    kanal {i+1}: WARUNKOWY (4-6/10)")
        else:
            print(f"    kanal {i+1}: MARGINALNY (<4/10)")

    print()
    print("=" * 84)
    print("  WERDYKT")
    print("=" * 84)

    n_stable = int((n_used >= 7).sum())
    n_marginal = int((n_used < 4).sum())

    if n_stable >= 4:
        print(f"  >>> MESH. Wszystkie {N_ENCODERS} kanalow stale.")
        print(f"      Sieć nie degeneruje do hierarchii — utrzymuje redundancje.")
    elif n_stable >= 2 and n_marginal >= 1:
        print(f"  >>> HIERARCHIA. {n_stable} stalych kanalow, {n_marginal} marginalnych.")
        print(f"      Ewolucja wybiera podzbior kanalow.")
    elif n_stable <= 1:
        print(f"  >>> DEGENERACJA. Tylko {n_stable} kanalow przetrwalo.")
        print(f"      Siec redukuje sie do pary — skalowanie nie dziala.")
    else:
        print(f"  >>> WYNIK POSREDNI. Sprawdz seedy.")