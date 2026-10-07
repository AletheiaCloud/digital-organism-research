# Digital Organism — Raport końcowy etapu RNN
## Trzy poziomy emergentnej organizacji i trzy zlokalizowane granice

**Data:** 2026-10-06
**Status:** Etap RNN zamknięty. 32 eksperymenty.

---

## STRESZCZENIE

Osiągnęliśmy **trzy poziomy emergentnej organizacji** w architekturze RNN
i **trzy zlokalizowane granice**. Jedna z granic została przełamana.

| Poziom | Funkcja | Kluczowa liczba |
|--------|---------|-----------------|
| 1 | 5 funkcji emergentnych | Pamięć, adaptacja, komunikacja, kompozycja, koordynacja |
| 2a | Self-model | R²(future) = 0.57 przy h32 |
| 2b | Other-model funkcjonalny | DELTA = +0.44, 9/10 seedów |
| 3 | Sieć 3 agentów | dBC = +0.57 (binarny kanał, 3D) |

**Trzy zlokalizowane granice:**

1. **Task dimension** — 3D wymaga wąskiego kanału (1 bit).
2. **Network size** — 3 agentów to sweet spot w 2D (niezależnie od kanału).
3. **Kombinacja** — 5 agentów w 3D = katastrofa.

---

## 1. POZIOM 1 — 5 funkcji emergentnych

Pięć funkcji emergentnych w monolit-RNN, każda przyczynowa i reprodukowalna:

| Funkcja | Siła dowodu |
|---------|-------------|
| Pamięć | +40.5% improvement, 5/5 seedów, memory gain +31% |
| Adaptacja | Tracking w każdym wariancie P (gładka zależność) |
| Komunikacja | 10/10 seedów, R²=0.78, random psuje bardziej niż mute |
| Kompozycja | alignment < 0.1 pod presją, 5/5 seedów |
| Koordynacja reaktywna | corr(g,opt) = +0.57 |

**Pliki:** `world.py`, `recurrent_v2.py`, `evolve.py`, `cooperation_v2.py`,
`cooperation_compose.py`, `role_coordination_v3.py`

---

## 2. POZIOM 2a — self-model

**Setup:** świat z opóźnionym wpływem stanu: `h(t+1) = A·h + B·a + C·s(t-5)`.

**Wynik:**

| hidden | R²(future) | ratio |
|--------|-----------|-------|
| 8 | 0.029 | 0.11 |
| 16 | 0.309 | 0.56 |
| 32 | **0.575** | 0.78 |

**Wniosek:** self-model jest **pojemnościowy** — większy monolit → lepszy.

**Pliki:** `self_delayed.py`, `scale_up.py`

---

## 3. POZIOM 2b — other-model (przełom)

### 3.1 Kontekst

**23 porażki** w świecie symetrycznym:

- self_delayed, meta_plasticity, modulated_v2, arch_test
- role_coordination v1-v4, graph_agents, no_arbiter
- graph_pure, metabolism v1/v2, cooldown
- interaction_memory, differential_selection
- competition, competition_v2, attention_other

**Wspólny wzorzec:** agenci zbiegali do punktu stałego (dominacja, synchronizacja, paraliż). Bez dynamiki — brak czego modelować.

### 3.2 Przełom — `coupled_tracking.py`

**Świat:** `z(t) ∈ ℝ²`, oscylacja (PERIOD=8, DECAY=0.95).
**Role:** A widzi 1D, B widzi 2D i wysyła 1D.

**Wynik:** DELTA = +0.435, 9/10 seedów.

**Uniwersalność (4/4 konfiguracji):**

| Config | PERIOD | DECAY | DELTA | #>0.05 |
|--------|--------|-------|-------|--------|
| baseline | 8 | 0.95 | +0.435 | 9/10 |
| fast | 5 | 0.95 | +0.480 | 10/10 |
| slow | 12 | 0.95 | +0.640 | 9/10 |
| fast_decay | 8 | 0.90 | +0.268 | 7/10 |

**Wszystkie 4 konfiguracje: SUKCES.**

### 3.3 Mechanizm

- B koduje ~1 bit o bieżącym `z(t)`.
- A dekoduje i łączy z własną obserwacją.
- Mutual information: ~1 bit.

**Pliki:** `coupled_tracking.py`, `coupled_tracking_v2.py`,
`generalization.py`, `message_analysis.py`

---

## 4. POZIOM 3 — sieć agentów

### 4.1 3-agent 2D (ciągły kanał)

| Metryka | Wartość |
|---------|---------|
| full R² | +0.404 |
| DELTA_B > 0.05 | 8/10 |
| DELTA_C > 0.05 | 4/10 |
| DELTA_both > 0.05 | 8/10 |
| superaddytywne | 6/10 |

**Mechanizm:** ensemble dwóch niezależnych reprezentacji tej samej informacji.
B i C kodują **tę samą oś** `z` (7/10 seedów), ale w **różny sposób**.

### 4.2 Test falsyfikacyjny — wymuszona redundancja

**Setup:** P_C = P_A (C widzi identycznie jak A).

**Wynik:** sieć NIE upadła — wzrosła (dBC = +1.036 vs +0.442).

**Interpretacja:** sieć jest **adaptacyjna**:
- Gdy obserwacje różne → C dostarcza uzupełniający wymiar.
- Gdy obserwacje identyczne → B staje się krytyczny, C staje się denoiserem.

### 4.3 Denoising test — sweep SIGMA_OBS_A

| sigma_A | dB | dC | dBC |
|---------|-----|-----|-----|
| 0.05 | +0.787 | +0.039 | +0.784 |
| 0.10 | +0.635 | +0.239 | +0.832 |
| 0.20 | +1.409 | **+0.716** | **+2.436** |
| 0.30 | +0.712 | +0.002 | +0.762 |

**Odkrycie:** nie monotoniczny — **sweet spot przy sigma_A = 0.20**.
Sieć jest **adaptacyjnym routerem** informacji.

**Pliki:** `coupled_tracking_3agents.py`, `network_analysis.py`,
`network_forced_redundancy.py`, `denoising_test.py`

---

## 5. GRANICE — trzy zlokalizowane

### 5.1 Task dimension (granica przełamana)

| Setup | full R² | dBC |
|-------|---------|-----|
| 3-agent 2D ciągły | +0.404 | +0.442 |
| 3-agent 3D ciągły H=1 | -0.200 | +0.076 |
| 3-agent 3D ciągły H=5 | -0.057 | -0.008 |
| 3-agent 3D H=32 | -0.243 | +0.040 |
| **3-agent 3D binarny H=5** | **+0.123** | **+0.568** |

**Wniosek:** granica 3D była **pozorna** — wynikała z szerokiego kanału, który RNN ignoruje. **Wąski kanał ratuje 3D.**

### 5.2 Network size (granica twarda)

| Setup | full R² |
|-------|---------|
| 3-agent 2D | +0.404 |
| 4-agent (? nieztestowane) | ? |
| **5-agent 2D ciągły** | **-0.444** |
| **5-agent 2D binarny** | **-0.594** |
| 5-agent 3D ciągły | -0.339 |

**Wniosek:** 3 agentów to sweet spot w 2D. 5 = za dużo.
**Wąski kanał NIE pomaga** w 5-agent (przeciążenie uwagi).

### 5.3 Reguła skalowania

| Zadanie \ Agenci | 3 | 5 |
|------------------|---|-----|
| 2D szeroki kanał | ✔ | ✗ |
| 2D wąski kanał | (nietestowane) | ✗ |
| 3D szeroki kanał | ✗ | (nietestowane) |
| **3D wąski kanał** | **✔** | (nietestowane) |

**Reguła:** wąski kanał pomaga **tylko** gdy zadanie jest trudne (3D)
i agentów jest mało (3).

---

## 6. PRZEŁOM — reguła wąskiego kanału

**Odkrycie:**

> RNN **ignoruje** szeroki kanał komunikacji. RNN **musi użyć** wąskiego kanału.

**Dowód:**

- Ciągły kanał (4 bajty/krok) → RNN ignoruje → dBC = 0, 0/10 seedów.
- Binarny kanał (1 bit/krok) → RNN używa → dBC = +0.57, **10/10 seedów**.

**Implikacja:** dla trudnych zadań (3D) — kanał **musi** być wąski,
żeby wymusić użycie.

---

## 7. MAPA PROJEKTU

| Poziom | Status |
|--------|--------|
| 1 — funkcje emergentne | ✔ |
| 2a — self-model | ✔ |
| 2b — other-model | ✔ |
| 3 — sieć agentów | ✔ (3-agent 2D i 3D binarny) |
| 4 — open-ended evolution | do zrobienia |
| 5 — metabolizm, samoreprodukcja | daleka przyszłość |

**Granice:**

- Network size: 3 w 2D (twarda).
- Task dimension: 3D wymaga wąskiego kanału (przełamana).
- Kombinacja: 5-agent 3D = katastrofa.

---

## 8. KIERUNEK DŁUGOTERMINOWY

Nie budujemy AGI. Budujemy **cyfrowy organizm**, który emerguje
z minimalnych reguł, bez wzorowania się na biologii.

**Zasada:** zaawansowane mechanizmy emergentne (self-model, other-model,
sieć, meta-reprezentacja, walencja) mogą — jeśli w ogóle — wyłonić to,
co nazwiemy świadomością. Bez projektowania, bez etykiet, bez definicji.

**Dystans:** 6 funkcji jakościowych:

| Funkcja | Status |
|---------|--------|
| Self-model + other-model razem | ✗ nigdy nie testowane razem |
| Trwały „ja" w czasie | ✗ |
| Meta-reprezentacja | ✗ |
| Walencja (organizm dba o siebie) | ✗ |
| Autonomia | ✗ |
| Metabolizm | ✗ |

**Każda wymaga nowego typu eksperymentu.** Każda może być nieosiągalna.

**Jesteśmy na drodze. Nie blisko — ale na drodze.**

---

## 9. CO DALEJ — open-ended evolution

**Nowy paradygmat.** Nie „optymalizuj funkcję" ale „generuj nowość".

**Zasady:**

- Selekcja przez **nowość**, nie przez fitness.
- Brak końcowego optimum.
- Ekosystem z wieloma niszami.
- Specjacja jako mechanizm.

**Cel:** sprawdzić, czy z minimalnych reguł mogą powstać
**nowe funkcje bez końca** — nie tylko jedna zoptymalizowana.

**Plik:** `open_evolution.py` (do napisania).

---

## 10. PLIKI PROJEKTU

**Poziom 1:** `world.py`, `recurrent_v2.py`, `evolve.py`,
`cooperation_v2.py`, `cooperation_compose.py`, `role_coordination_v3.py`

**Poziom 2a:** `self_delayed.py`, `scale_up.py`

**Poziom 2b:** `coupled_tracking.py`, `coupled_tracking_v2.py`,
`generalization.py`, `message_analysis.py`

**Poziom 3:** `coupled_tracking_3agents.py`, `network_analysis.py`,
`network_forced_redundancy.py`, `denoising_test.py`

**Przełom:** `coupled_tracking_3agents_3d_1bit.py`

**Granice:** `coupled_tracking_5agents.py`, `coupled_tracking_5agents_3d.py`,
`coupled_tracking_5agents_1bit.py`

---

## 11. STATYSTYKA PROJEKTU

- **32 eksperymenty**
- **3 poziomy osiągnięte**
- **3 granice zlokalizowane**
- **1 granica przełamana**
- **0 obietnic AGI**

**Metodologia:**

- Falsyfikacja przed entuzjazmem.
- Wiele seedów (min. 5–10).
- Baseline zawsze obecny.
- Rozdzielenie `structure_seed` od `noise_seed`.
- Ablacje zamiast porównań.

---

*Koniec raportu. Stan na 2026-10-06.*