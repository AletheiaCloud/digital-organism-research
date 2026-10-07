"""
COUPLED_TRACKING v2: prawdziwa persystencja + ablacja wiadomosci.
Kluczowa metryka: DELTA = R2_A_full - R2_A_mute.
Jesli DELTA > 0.05 => A UZYWA wiadomosci od B.
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
        self.P_A = rng.standard_normal((1, 2))
        self.P_A /= np.linalg.norm(self.P_A)
        self.t = 0
    def observe_A(self):
        return self.P_A @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def observe_B(self):
        return self.z + self.rng.standard_normal(2) * SIGMA_OBS
    def step(self):
        self.z = self.A @ self.z + self.rng.standard_normal(2) * SIGMA_Z
        self.t += 1
    def get_z(self):
        return self.z.copy()

# ============================================================
# AGENT
# ============================================================
class Agent:
    def __init__(self, role, m_own, rng):
        self.role = role
        self.m_own = m_own
        m_in = m_own + (1 if role == "A" else 0)
        sc = 0.1
        self.W_s = rng.standard_normal((HIDDEN, HIDDEN)) * sc
        self.W_o = rng.standard_normal((HIDDEN, m_in)) * sc
        self.b_s = np.zeros(HIDDEN)
        self.W_pred = rng.standard_normal((2, HIDDEN)) * sc
        self.b_pred = np.zeros(2)
        if role == "B":
            self.W_msg = rng.standard_normal((1, HIDDEN)) * sc
            self.b_msg = np.zeros(1)
    def init_state(self):
        return np.zeros(HIDDEN)
    def step(self, s, obs, msg_in=0.0):
        if self.role == "A":
            x = np.concatenate([obs, [msg_in]])
        else:
            x = obs
        s_new = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        z_pred = self.W_pred @ s_new + self.b_pred
        if self.role == "B":
            msg = float(np.tanh(self.W_msg @ s_new + self.b_msg)[0])
            return z_pred, msg, s_new
        return z_pred, s_new

# ============================================================
# EPIZOD
# ============================================================
def episode(agA, agB, w, mute=False, record=False):
    sA = agA.init_state()
    sB = agB.init_state()
    prev_msg_B = 0.0
    rA_tot = 0.0
    rB_tot = 0.0
    z_preds_A = []
    z_preds_B = []
    z_trues = []
    z_ats = []
    msgs = []
    for _ in range(T):
        z_now = w.get_z()
        oA = w.observe_A()
        oB = w.observe_B()
        msg_to_A = 0.0 if mute else prev_msg_B
        z_pred_A, sA = agA.step(sA, oA, msg_to_A)
        z_pred_B, msg_B, sB = agB.step(sB, oB)
        w.step()
        z_next = w.get_z()
        rA_tot -= float(np.sum((z_pred_A - z_next) ** 2))
        rB_tot -= float(np.sum((z_pred_B - z_next) ** 2))
        z_preds_A.append(z_pred_A.copy())
        z_preds_B.append(z_pred_B.copy())
        z_trues.append(z_next.copy())
        z_ats.append(z_now.copy())
        msgs.append(msg_B)
        prev_msg_B = msg_B
    if record:
        return (rA_tot, rB_tot, np.array(z_preds_A), np.array(z_preds_B),
                np.array(z_trues), np.array(z_ats), np.array(msgs))
    return rA_tot, rB_tot, None, None, None, None, None

# ============================================================
# METRYKI
# ============================================================
def r2_raw(pred, actual):
    """R2 bez regresji — surowa predykcja vs rzeczywistosc."""
    mse = np.mean(np.sum((pred - actual) ** 2, axis=1))
    var = np.mean(np.sum((actual - actual.mean(axis=0)) ** 2, axis=1))
    return float(1.0 - mse / (var + 1e-12))

def r2_true_persistence(z_at_t, z_true):
    """Prawdziwa persystencja: z(t+1) = z(t)."""
    return r2_raw(z_at_t, z_true)

# ============================================================
# EWALUACJA
# ============================================================
def evaluate_pair(agA, agB, seeds):
    total = 0.0
    for s in seeds:
        w = OscWorld(s)
        rA, rB, _, _, _, _, _ = episode(agA, agB, w)
        total += rA + rB
    return total / len(seeds)

def mutate(ag, rng, sigma):
    c = copy.deepcopy(ag)
    for attr in ["W_s", "W_o", "b_s", "W_pred", "b_pred"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if ag.role == "B":
        for attr in ["W_msg", "b_msg"]:
            v = getattr(c, attr)
            setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    return c

def evolve(seed, N=N_POP, G=G):
    rng = np.random.default_rng(seed)
    def make_pair():
        return [Agent("A", 1, rng), Agent("B", 2, rng)]
    pop = [make_pair() for _ in range(N)]
    best_fit = -np.inf
    best = None
    for gen in range(G):
        fits = np.array([evaluate_pair(t[0], t[1], TRAIN_SEEDS) for t in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]
        fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]
            best = [copy.deepcopy(pop[0][0]), copy.deepcopy(pop[0][1])]
        n_el = max(1, int(N * 0.25))
        elite = pop[:n_el]
        children = []
        while len(children) < N - n_el:
            p = elite[rng.integers(n_el)]
            children.append([mutate(p[0], rng, 0.05),
                             mutate(p[1], rng, 0.05)])
        pop = elite + children
    return best, best_fit

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 84)
    print("  COUPLED_TRACKING v2: prawdziwa persystencja + ablacja")
    print("=" * 84)
    print(f"  PERIOD={PERIOD}  DECAY={DECAY}  HIDDEN={HIDDEN}")
    print(f"  Kluczowa metryka: DELTA = R2_A_full - R2_A_mute")
    print()

    results = []
    for seed in SEEDS:
        best, bf = evolve(seed, G=G)
        agA, agB = best
        # FULL
        w = OscWorld(TEST_SEEDS[0])
        _, _, z_pred_A, z_pred_B, z_true, z_at_t, msgs = episode(
            agA, agB, w, mute=False, record=True)
        r2_A_full = r2_raw(z_pred_A, z_true)
        r2_B_full = r2_raw(z_pred_B, z_true)
        # MUTE
        w = OscWorld(TEST_SEEDS[0])
        _, _, z_pred_A_m, _, z_true_m, _, _ = episode(
            agA, agB, w, mute=True, record=True)
        r2_A_mute = r2_raw(z_pred_A_m, z_true_m)
        # baseline
        r2_pers = r2_true_persistence(z_at_t, z_true)

        delta_A = r2_A_full - r2_A_mute

        # korelacja wiadomosci z prawdziwym z
        if np.std(msgs) > 1e-6:
            msg_corr_x = float(np.corrcoef(msgs, z_true[:, 0])[0, 1])
            msg_corr_y = float(np.corrcoef(msgs, z_true[:, 1])[0, 1])
        else:
            msg_corr_x = msg_corr_y = 0.0

        results.append((seed, r2_A_full, r2_A_mute, delta_A, r2_B_full,
                        r2_pers, msg_corr_x, msg_corr_y))
        print(f"  seed={seed:>2}: R2_A_full={r2_A_full:+.3f}  "
              f"R2_A_mute={r2_A_mute:+.3f}  DELTA={delta_A:+.3f}  "
              f"R2_B={r2_B_full:+.3f}  R2_pers={r2_pers:+.3f}  "
              f"msg_corr=({msg_corr_x:+.2f},{msg_corr_y:+.2f})")

    print()
    print("=" * 84)
    print("  PODSUMOWANIE")
    print("=" * 84)
    d = np.array([r[3] for r in results])
    pers = np.array([r[5] for r in results])
    full = np.array([r[1] for r in results])
    mute = np.array([r[2] for r in results])
    r2B = np.array([r[4] for r in results])

    print(f"  R2_A_full:  mean={full.mean():+.3f}  min={full.min():+.3f}  max={full.max():+.3f}")
    print(f"  R2_A_mute:  mean={mute.mean():+.3f}  min={mute.min():+.3f}  max={mute.max():+.3f}")
    print(f"  R2_B:       mean={r2B.mean():+.3f}")
    print(f"  R2_persist: mean={pers.mean():+.3f}")
    print(f"  DELTA (full-mute): mean={d.mean():+.3f}  min={d.min():+.3f}  max={d.max():+.3f}")
    print()
    print(f"  seedow z DELTA > 0.05: {int((d > 0.05).sum())}/10")
    print(f"  seedow z DELTA > 0.02: {int((d > 0.02).sum())}/10")
    print(f"  seedow z DELTA > 0.00: {int((d > 0.00).sum())}/10")
    print()

    if (d > 0.05).sum() >= 7:
        print("  >>> SUKCES. A UZYWA wiadomosci od B (DELTA > 0.05 w >=7/10).")
        print("      Other-model emergentny, przyczynowo istotny.")
    elif (d > 0.02).sum() >= 5:
        print("  >>> CZESCIOWY SUKCES. Sygnal jest (DELTA > 0.02 w >=5/10).")
        print("      Kierunek dobry, wymaga wzmocnienia.")
    elif (d > 0.0).sum() >= 5:
        print("  >>> SLABY. Kierunek dobry, sily brak.")
        print("      A uzywa wiadomosci marginalnie.")
    else:
        print("  >>> PORAZKA. A ignoruje wiadomosc od B.")
        print("      Other-model nie emergowal nawet z jawna presja.")