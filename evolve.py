"""
Reproducibility test: 5 niezależnych seedów ewolucji.
Wszystko w jednym pliku, brak importów z lokalnych modułów.
"""
import numpy as np
import copy
from world import World
from organism import concat_obs


STRUCT_SEED = 1
TRAIN_NOISE = list(range(1, 21))
TEST_NOISE  = list(range(1000, 1021))
T = 100
SIGMAS = (0.10, 0.25, 0.05)
ACTION_DELAY = 0
N_STATE = 8
M1, M2, M3 = 2, 1, 1
M_IN = M1 + M2 + M3


class RecOrg:
    def __init__(self, m_in, k, hidden, rng):
        self.m_in, self.k, self.hidden = m_in, k, hidden
        sc = 0.1
        self.W_s = rng.standard_normal((hidden, hidden)) * sc
        self.W_o = rng.standard_normal((hidden, m_in)) * sc
        self.b_s = np.zeros(hidden)
        self.W_a = rng.standard_normal((k, hidden)) * sc
        self.b_a = np.zeros(k)
    def init_state(self): return np.zeros(self.hidden)
    def step(self, s, o):
        s2 = np.tanh(self.W_s @ s + self.W_o @ o + self.b_s)
        return np.tanh(self.W_a @ s2 + self.b_a), s2


def _world(noise_seed):
    return World(n=N_STATE, m1=M1, m2=M2, m3=M3,
                 structure_seed=STRUCT_SEED, noise_seed=noise_seed,
                 sigmas=SIGMAS, action_delay=ACTION_DELAY)


def run(org, noise_seed, ablate=False):
    w = _world(noise_seed)
    s = org.init_state()
    r = 0.0
    for _ in range(T):
        o1, o2, o3 = w.observe()
        a, s = org.step(s, concat_obs(o1, o2, o3))
        if ablate: s = org.init_state()
        r += w.step(a)
    return r


def run_zero(noise_seed):
    w = _world(noise_seed)
    r = 0.0
    for _ in range(T):
        r += w.step(np.zeros(1))
    return r


def fitness(org, noise_list, ablate=False):
    return float(np.mean([run(org, ns, ablate) for ns in noise_list]))


def mutate(org, rng, sigma):
    c = copy.deepcopy(org)
    c.W_s += rng.standard_normal(c.W_s.shape) * sigma
    c.W_o += rng.standard_normal(c.W_o.shape) * sigma
    c.b_s += rng.standard_normal(c.b_s.shape) * sigma
    c.W_a += rng.standard_normal(c.W_a.shape) * sigma
    c.b_a += rng.standard_normal(c.b_a.shape) * sigma
    return c


def evolve(seed, N=40, G=200, hidden=8, elite=0.25, sigma=0.05):
    rng = np.random.default_rng(seed)
    pop = [RecOrg(M_IN, 1, hidden, rng) for _ in range(N)]
    best_ever = None; best_fit = -np.inf
    for gen in range(G):
        fits = np.array([fitness(o, TRAIN_NOISE) for o in pop])
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]; fits = fits[idx]
        if fits[0] > best_fit:
            best_fit = fits[0]; best_ever = copy.deepcopy(pop[0])
        n_elite = max(1, int(N * elite))
        elite_pop = pop[:n_elite]
        children = []
        while len(children) < N - n_elite:
            p = elite_pop[rng.integers(n_elite)]
            children.append(mutate(p, rng, sigma))
        pop = elite_pop + children
    return best_ever


if __name__ == "__main__":
    print("=" * 70)
    print("  REPRODUCIBILITY: 5 niezaleznych seedow ewolucji")
    print("=" * 70)

    base_train = float(np.mean([run_zero(ns) for ns in TRAIN_NOISE]))
    base_test  = float(np.mean([run_zero(ns) for ns in TEST_NOISE]))
    print(f"  baseline train={base_train:.3f}  test={base_test:.3f}")
    print()

    results = []
    for s in [1, 2, 3, 4, 5]:
        print(f"--- seed={s} start ---")
        org = evolve(seed=s)
        tr  = fitness(org, TRAIN_NOISE)
        te  = fitness(org, TEST_NOISE)
        atr = fitness(org, TRAIN_NOISE, ablate=True)
        ate = fitness(org, TEST_NOISE, ablate=True)
        imp_te  = (te - base_test) / abs(base_test) * 100
        gain_te = (te - ate) / abs(ate) * 100
        print(f"    test_fit={te:.3f}  improvement={imp_te:+.1f}%  memory_gain={gain_te:+.1f}%")
        results.append((s, te, imp_te, gain_te))

    print()
    print("=" * 70)
    print("  PODSUMOWANIE")
    print("=" * 70)
    print(f"  {'seed':>6}  {'test_fit':>12}  {'improv%':>10}  {'mem_gain%':>12}")
    print("  " + "-" * 46)
    for s, te, imp, g in results:
        print(f"  {s:>6}  {te:>12.3f}  {imp:>+9.1f}%  {g:>+11.1f}%")
    print()

    gains = np.array([g for _, _, _, g in results])
    imps  = np.array([imp for _, _, imp, _ in results])
    print(f"  memory_gain:  mean={gains.mean():+.1f}%  std={gains.std():.1f}%")
    print(f"  improvement:  mean={imps.mean():+.1f}%  std={imps.std():.1f}%")
    print(f"  seedow z memory_gain > 10%: {int((gains > 10).sum())} / 5")
    print(f"  seedow z improvement > 20%: {int((imps > 20).sum())} / 5")
    print()

    if (gains > 10).sum() >= 4:
        print("  [VERDICT] REPRODUCIBLE.")
    elif (gains > 10).sum() >= 2:
        print("  [VERDICT] PARTIALLY REPRODUCIBLE.")
    else:
        print("  [VERDICT] NOT REPRODUCIBLE.")