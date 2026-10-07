# Digital Organism — Stage 1: Empirical Limits of Monolithic RNNs in Multi-Agent Social Modeling

**Author:** Tomasz Rojek (AletheiaCloud) · Independent researcher, Poland
**Status:** Stage 1 closed (2026-10-06) — 32 experiments, 3 levels of emergent organization, 3 localized architectural boundaries, 1 boundary broken.
**Archived snapshot (DOI):** [Zenodo v1.0.0] https://doi.org/10.5281/zenodo.25205500

> We are not building AGI. We are building a digital organism that emerges from minimal rules, without being modeled on biology.

## Summary

| Level | Function | Key number |
|---|---|---|
| 1 | 5 emergent functions | memory, adaptation, semantic communication, composition, reactive coordination |
| 2a | Self-model | R²(future) = 0.57 at h32 (baseline h8: 0.03) |
| 2b | Functional other-model | DELTA = +0.44, 9/10 seeds |
| 3 | 3-agent network | dBC = +0.57 (binary channel, 3D) |

**Localized boundaries:** Task dimension (3D requires a narrow channel — *broken*), network size (3 agents = sweet spot in 2D), combination (5 agents in 3D = collapse).

## The Narrow-Channel Rule (key finding)

Monolithic RNNs **ignore wide continuous channels** and only use **narrow (1-bit) bottlenecks**:

| Channel | Result |
|---|---|
| Continuous (4 B/step) | ignored → dBC ≈ 0, 0/10 seeds |
| Binary (1 bit/step) | used → dBC = +0.57, 10/10 seeds |

Implication: for hard tasks (3D) the channel must be narrow to force usage — a structural constraint on emergent communication in recurrent architectures.

## Why monoliths fail at class 2b (social modeling)

17 experiments, one pattern: under evolutionary pressure, monolithic RNNs always converge to a **fixed point** — dominance, synchronization, or paralysis — never alternation (a cycle, not a fixed point). Independent of capacity (h8→h32), structure (monolith/graph/separate organisms), pressure (competition/cooperation/metabolism/cooldown), and communication. Class 2a (temporal self-model) scales with capacity; class 2b (social other-model) gets **worse** with scale (h32 < h8) — a structural, not capacity, limit.

## Repository structure

| File | Evidence |
|---|---|
| `world.py` | environment: hidden state, partial observability, `structure_seed`/`noise_seed` split |
| `evolve.py` | evolution engine + Class-1 memory experiments |
| `cooperation_v2.py` | semantic communication (R² = 0.78; random hurts more than mute) |
| `scale_up.py` | Class-2a self-model scaling (h8→h32) and Class-2b scaling failure |
| `coupled_tracking_v2.py` | Class-2b breakthrough: functional other-model (DELTA = +0.44, 9/10) |
| `coupled_tracking_3agents_3d_1bit.py` | narrow-channel breakthrough in 3D (dBC = +0.57) |
| `coupled_tracking_5agents.py` / `_1bit.py` | hard boundary: 5 agents = collapse; narrow channel does not rescue |
| `REPORT_FINAL.md` | full report: 32 experiments, methodology, falsification tests |

## Methodology

Falsification before enthusiasm · multiple seeds (5–10) with train/test split · baseline always present (incl. true-persistence baseline) · `structure_seed` separated from `noise_seed` · ablations over comparisons.

## Licensing & usage

- **Code:** AGPL-3.0 — derivative work must remain open-source and attribute this repository.
- **Reports & data:** CC BY-NC-ND 4.0 — cite with attribution; no alteration or direct commercial use without permission.
