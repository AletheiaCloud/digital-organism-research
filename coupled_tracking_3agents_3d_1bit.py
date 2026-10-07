"""
COUPLED_TRACKING 3 AGENTS 3D BINARY: kanal binarny (1 bit na wiadomosc).
B i C wysylaja tanh(5x) - praktycznie {-1, +1}.
HORIZON=5 - bezposrednie porownanie z poprzednim eksperymentem.
"""
import numpy as np
import copy

# ============================================================
# PARAMETRY
# ============================================================
N_STATE = 3
PERIOD = 8
DECAY = 0.95
SIGMA_Z = 0.05
SIGMA_OBS = 0.1
T = 100
HORIZON = 5
HIDDEN = 16
N_POP = 30
G = 100
SEEDS = list(range(1, 11))
TRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(100, 106))

SATURATION = 5.0

def rotation_matrix_3d(theta, axis):
    c = np.cos(theta); s = np.sin(theta)
    if axis == 0:
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R

class OscWorld3D:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        theta = 2 * np.pi / PERIOD
        axis = int(rng.integers(0, 3))
        R = rotation_matrix_3d(theta, axis)
        self.A = R * DECAY
        self.rng = rng
        self.z = rng.standard_normal(N_STATE) * 0.5
        self.P_A = rng.standard_normal((1, N_STATE))
        self.P_A /= np.linalg.norm(self.P_A)
        self.P_C = rng.standard_normal((1, N_STATE))
        self.P_C /= np.linalg.norm(self.P_C)
        self.t = 0
    def observe_A(self):
        return self.P_A @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def observe_B(self):
        return self.z + self.rng.standard_normal(N_STATE) * SIGMA_OBS
    def observe_C(self):
        return self.P_C @ self.z + self.rng.standard_normal(1) * SIGMA_OBS
    def step(self):
        self.z = self.A @ self.z + self.rng.standard_normal(N_STATE) * SIGMA_Z
        self.t += 1
    def get_z(self):
        return self.z.copy()

class Agent:
    def __init__(self, role, m_own, rng):
        self.role = role
        self.m_own = m_own
        if role == "A":
            m_in = m_own + 2
        else:
            m_in = m_own
        self.m_in = m_in
        sc = 0.1
        self.W_s = rng.standard_normal((HIDDEN, HIDDEN)) * sc
        self.W_o = rng.standard_normal((HIDDEN, m_in)) * sc
        self.b_s = np.zeros(HIDDEN)
        self.W_pred = rng.standard_normal((N_STATE, HIDDEN)) * sc
        self.b_pred = np.zeros(N_STATE)
        if role in ["B", "C"]:
            self.W_msg = rng.standard_normal((1, HIDDEN)) * sc
            self.b_msg = np.zeros(1)
    def init_state(self):
        return np.zeros(HIDDEN)
    def step(self, s, obs, msg_in_1=0.0, msg_in_2=0.0):
        if self.role == "A":
            x = np.concatenate([obs, [msg_in_1, msg_in_2]])
        else:
            x = obs
        s_new = np.tanh(self.W_s @ s + self.W_o @ x + self.b_s)
        z_pred = self.W_pred @ s_new + self.b_pred
        if self.role in ["B", "C"]:
            # FIX: uzyj [0] zeby wyciagnac skalar
            raw = float((self.W_msg @ s_new + self.b_msg)[0])
            msg = float(np.tanh(SATURATION * raw))
            return z_pred, msg, s_new
        return z_pred, s_new

def episode(agA, agB, agC, w, mute_B=False, mute_C=False, record=False):
    sA = agA.init_state()
    sB = agB.init_state()
    sC = agC.init_state()
    prev_msg_B = 0.0
    prev_msg_C = 0.0
    rA_tot = 0.0
    rB_tot = 0.0
    rC_tot = 0.0

    z_at_t_list = []
    z_pred_A_list = []
    pred_buffer_A = []

    for t in range(T):
        z_now = w.get_z()
        z_at_t_list.append(z_now.copy())

        oA = w.observe_A()
        oB = w.observe_B()
        oC = w.observe_C()
        msg_to_A_B = 0.0 if mute_B else prev_msg_B
        msg_to_A_C = 0.0 if mute_C else prev_msg_C

        z_pred_A, sA = agA.step(sA, oA, msg_to_A_B, msg_to_A_C)
        z_pred_B, msg_B, sB = agB.step(sB, oB)
        z_pred_C, msg_C, sC = agC.step(sC, oC)

        z_pred_A_list.append(z_pred_A.copy())
        pred_buffer_A.append((t, z_pred_A.copy()))

        if len(pred_buffer_A) > HORIZON:
            _, z_pred_old = pred_buffer_A.pop(0)
            rA_tot -= float(np.sum((z_pred_old - z_now) ** 2))

        prev_msg_B = msg_B
        prev_msg_C = msg_C
        w.step()

    if record:
        z_pred_arr = np.array(z_pred_A_list)[:T - HORIZON]
        z_true_arr = np.array(z_at_t_list)[HORIZON:]
        z_at_t_arr = np.array(z_at_t_list)[:T - HORIZON]
        return rA_tot, rB_tot, rC_tot, z_pred_arr, z_true_arr, z_at_t_arr
    return rA_tot, rB_tot, rC_tot, None, None, None

def r2_raw(pred, actual):
    mse = np.mean(np.sum((pred - actual) ** 2, axis=1))
    var = np.mean(np.sum((actual - actual.mean(axis=0)) ** 2, axis=1))
    return float(1.0 - mse / (var + 1e-12))

def evaluate_trio(agA, agB, agC, seeds):
    total = 0.0
    for s in seeds:
        w = OscWorld3D(s)
        rA, rB, rC, _, _, _ = episode(agA, agB, agC, w)
        total += rA + rB + rC
    return total / len(seeds)

def mutate(ag, rng, sigma):
    c = copy.deepcopy(ag)
    for attr in ["W_s", "W_o", "b_s", "W_pred", "b_pred"]:
        v = getattr(c, attr)
        setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    if ag.role in ["B", "C"]:
        for attr in ["W_msg", "b_msg"]:
            v = getattr(c, attr)
            setattr(c, attr, v + rng.standard_normal(v.shape) * sigma)
    return c

def evolve(seed, N=N_POP, G=G):
    rng = np.random.default_rng(seed)
    def make_trio():
        return [Agent("A", 1, rng), Agent("B", N_STATE, rng), Agent("C", 1, rng)]
    pop = [make_trio() for _ in range(N)]
    best_fit = -np.inf; best = None
    for gen in range(G):
        fits = np.array([evaluate_trio(t[0], t[1], t[2], TRAIN_SEEDS) for t in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]; fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]
            best = [copy.deepcopy(pop[0][0]), copy.deepcopy(pop[0][1]), copy.deepcopy(pop[0][2])]
        n_el = max(1, int(N * 0.25))
        elite = pop[:n_el]
        children = []
        while len(children) < N - n_el:
            p = elite[rng.integers(n_el)]
            children.append([mutate(p[0], rng, 0.05), mutate(p[1], rng, 0.05), mutate(p[2], rng, 0.05)])
        pop = elite + children
    return best, best_fit

if __name__ == "__main__":
    print("=" * 84)
    print(f"  COUPLED_TRACKING 3 AGENTS 3D BINARY (H={HORIZON})")
    print("=" * 84)
    print(f"  Kanality binarne: tanh({SATURATION}*x) -> praktycznie {{-1, +1}}")
    print(f"  Test: czy agent uzywa komunikacji przy waskim kanale?")
    print()

    results = []
    for seed in SEEDS:
        best, bf = evolve(seed, G=G)
        agA, agB, agC = best
        w = OscWorld3D(TEST_SEEDS[0])
        _, _, _, z_pred_A, z_true, z_at_t = episode(agA, agB, agC, w, record=True)
        r2_full = r2_raw(z_pred_A, z_true)
        r2_pers = r2_raw(z_at_t, z_true)
        w = OscWorld3D(TEST_SEEDS[0])
        _, _, _, z_pred_mB, z_true_mB, _ = episode(agA, agB, agC, w, mute_B=True, record=True)
        r2_muteB = r2_raw(z_pred_mB, z_true_mB)
        w = OscWorld3D(TEST_SEEDS[0])
        _, _, _, z_pred_mC, z_true_mC, _ = episode(agA, agB, agC, w, mute_C=True, record=True)
        r2_muteC = r2_raw(z_pred_mC, z_true_mC)
        w = OscWorld3D(TEST_SEEDS[0])
        _, _, _, z_pred_mBC, z_true_mBC, _ = episode(agA, agB, agC, w, mute_B=True, mute_C=True, record=True)
        r2_muteBC = r2_raw(z_pred_mBC, z_true_mBC)

        dB = r2_full - r2_muteB
        dC = r2_full - r2_muteC
        dBC = r2_full - r2_muteBC
        results.append((seed, r2_full, dB, dC, dBC, r2_pers))
        print(f"  seed={seed:>2}: full={r2_full:+.3f}  dB={dB:+.3f}  dC={dC:+.3f}  "
              f"dBC={dBC:+.3f}  pers={r2_pers:+.3f}")

    print()
    print("=" * 84)
    print("  PODSUMOWANIE")
    print("=" * 84)
    full = np.array([r[1] for r in results])
    dB = np.array([r[2] for r in results])
    dC = np.array([r[3] for r in results])
    dBC = np.array([r[4] for r in results])
    pers = np.array([r[5] for r in results])
    print(f"  full R2:    mean={full.mean():+.3f}  min={full.min():+.3f}  max={full.max():+.3f}")
    print(f"  DELTA_B:    mean={dB.mean():+.3f}")
    print(f"  DELTA_C:    mean={dC.mean():+.3f}")
    print(f"  DELTA_both: mean={dBC.mean():+.3f}")
    print(f"  pers:       mean={pers.mean():+.3f}")
    print()
    print(f"  seedow DELTA_both > 0.05: {int((dBC > 0.05).sum())}/10")
    print()

    print("=" * 84)
    print(f"  POROWNANIE Z CIAGLYM KANALEM (H={HORIZON})")
    print("=" * 84)
    print(f"  Ciagly H=5:  full=-0.057, dBC=-0.008, dBC>0.05 w 0/10")
    print(f"  Binary H=5:  full={full.mean():+.3f}, dBC={dBC.mean():+.3f}, "
          f"dBC>0.05 w {int((dBC > 0.05).sum())}/10")
    print()
    print("=" * 84)
    print("  WERDYKT")
    print("=" * 84)
    if (dBC > 0.05).sum() >= 7:
        print(f"  >>> PRZELOM. Kanaly binarne UZYWANE (dBC>0.05 w "
              f"{int((dBC > 0.05).sum())}/10).")
        print(f"      Wasciwy kanal wymusza komunikacje. RNN potrafi.")
    elif (dBC > 0.05).sum() >= 4:
        print(f"  >>> CZESCIOWY SUKCES. dBC>0.05 w {int((dBC > 0.05).sum())}/10.")
        print(f"      Wasciwy kanal pomaga, ale nie we wszystkich seedach.")
    else:
        print(f"  >>> WASKI KANAL NIE POMAGA. dBC>0.05 w "
              f"{int((dBC > 0.05).sum())}/10.")
        print(f"      Nawet gdy kanal jest waski, RNN nie uczy sie uzywac go.")
        print(f"      DEFINITYWNA GRANICA RNN w 3D.")