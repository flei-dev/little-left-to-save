# Pre-registration: main experiments of the study (English translation)

This is an English translation of `PREREG_C.md`, which is the governing text. The Chinese file was written on 2026-11-11 at 10:20 (UTC+8), before the server runs started. `PREREG_C_frozen_2026-11-11.md` is the file exactly as it was frozen (SHA-256 `52501f412426a8e7a434fdec85fcdf93053b4869a07ae2e60089313217a62704`). `PREREG_C.md` adds only the revision log at the end.

Working title at registration: *What Can Adaptive Streaming Still Save on Modern Smartphones? A Measurement-Calibrated, Oracle-Bounded Study of Long-Form and Short-Video Streaming.*

After this file was written, code and parameters were not changed. If a bug was found, the fix was to be followed by re-running every affected stage, with the time, reason and scope recorded in the revision log. SHA-256 hashes of code and data are in `MANIFEST.txt` of the experiment package.

## 1. Data and splits

### Content
- Library of 46 videos encoded with x264 (6 rungs, 300–4300 kbps, 240p–1080p).
- Each segment has its size, phone-model VMAF and standard VMAF.
- Main experiments use 4-s segments; E5 uses 2-s segments; E4 uses 2-s segments plus an AV1 encoding (SVT-AV1 preset 8, same 6 bitrates).

### Short-video feed
- Viewing behavior from KuaiRand-Pure, tab = 1.
  - Retention curves (visible to the player) are estimated from the first log only.
  - Sessions come from the second log only: 812 sessions, each with 20 consecutive views.
- Development: even-numbered sessions + LTE training traces. Used only to debug code, not to tune method parameters.
- Main evaluation: odd-numbered sessions (406) + LTE test traces (126 segments of 300 s).
  - Each trace is paired with 3 sessions: 378 evaluation sessions.
  - Random-seed base 9000, a pairing different from the earlier SPARE study.
- Note: the odd-numbered sessions were used in the earlier SPARE study. All parameter grids of this study are fixed in this file, and no parameter was tuned on these sessions.

### Long-form video
- Non-overlapping 300-s windows from each video, at most 2 per video: 66 windows.
- The viewer watches the whole window. The player buffer cap is a method parameter (30, 60 or 120 s).
- Main evaluation (E1/E2/E5): all 66 windows; each LTE test trace is paired with 2 windows: 252 evaluation sessions, seed base 9100.
- E3 (DRL) trains only on training videos (genre-stratified folds 2–4, 41 windows) and evaluates only on held-out videos (folds 0–1, 25 windows).

### Network traces
| Name | Source | Train / test segments | Median test throughput |
|---|---|---|---|
| LTE | Irish 4G (MMSys'18), 300-s windows | 235 / 126 | 10.4 Mbps |
| 5G | Irish 5G (MMSys'20), 300-s windows | 36 / 46 | 68 Mbps |
| old | earlier mixed set (3G / FCC / synthetic) | 213 / 92 | 3.2 Mbps |

## 2. Energy models

### Formula
Device energy is the sum of four parts:
- Radio: state machine (promotion → transfer → tail → idle). Power during transfer = β + α·throughput.
- Playback: p_base · watched seconds + c · decoded MB + extra decoding power · watched seconds.
- Stall: p_stall · stalled seconds.

### E1 grid (75 points)
Playback power fixed at p_base = p_stall = 1.10 W (COCONUT), α = 0. The other three parameters take:
- Energy per decoded MB c ∈ {0.0444, 0.1, 0.2, 0.4, 0.65} J/MB.
- Post-transfer tail (T, P) ∈ {(0, 0), (1 s, 0.25 W), (5 s, 1.06 W), (11.6 s, 1.06 W), (20 s, 2.12 W)}.
  - Points with T ≤ 1 s: no promotion delay, zero idle power (as measured).
  - Other points: 0.26-s promotion, with power proportional to the tail power.
- Transfer base power β ∈ {0.65 (measured), 1.29 (Huang'12 LTE), 2.58} W.

### Four anchors
| Name | Device | Radio | Note |
|---|---|---|---|
| meas | modern phone | measured LTE | equals grid point c = 0.0444, T = 1, β = 0.65 |
| modern_lte | modern phone | literature LTE | |
| modern_5g | modern phone | literature-derived 5G NSA | 20-s tail, doubled powers |
| legacy_lte | Galaxy S4 | literature LTE | setting of most prior work |

### E4: extra AV1 decoding power (relative to hardware H.264, at 1080p)
| Setting | Extra power | Basis |
|---|---|---|
| hardware | 0 | |
| dav1d | 0.115 W | measured on our phone |
| system decoder | 0.69 W | measured on our phone |
| COCONUT software decoding | 1.43 W | COCONUT |

- Main setting: extra power proportional to resolution (pixel count).
- Sensitivity: independent of resolution (flat).
- Radio: measured LTE (main); literature LTE (sensitivity).

## 3. Method grids (code `svs/grids.py`, fixed in advance)

λ grid = {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4}.

### Short-video feed
- Energy-unaware: Fixed (16 points), Dashlet (12), PDAS (9).
- Energy-aware: DashletE (2 × λ grid), BurstE (6 × λ grid), SPARE (λ grid).
- Clairvoyant reference (knows each view's watch time and the next 2 s of throughput): O-SPARE (nbar ∈ {1, 2, 4, 8} × λ grid), O-Burst (6 × λ grid).
- E2 decomposition: Ow-* (knows watch time only) and Ot-* (knows throughput only), same grids as O-*.

### Long-form video
Buffer cap B ∈ {30, 60, 120} s.
- Energy-unaware: Rate (safety factor {0.6, 0.8, 0.9, 1.0} × B), BBA ((reservoir, cushion) ∈ {(5, 10), (10, 20), (5, 30)} × B), BOLA (γp ∈ {1, 2, 5, 10} × B), MPC (one point per B).
- Energy-aware: MPC-E (λ > 0 × B), BurstMPC ((L, H) ∈ {(10, 60), (30, 60), (20, 120), (60, 120)} × λ grid).
- Clairvoyant reference (simulates each plan with the true future throughput): O-MPC (λ grid × B), O-BurstMPC (4 (L, H) × λ grid).

### About the clairvoyant reference
It is a reference value only. The paper calls it a "clairvoyant reference", not an optimum.

## 4. Main metrics

### Computation
- For each energy model (and, in E4, each codec mode), compute per parameter point the mean QoE per minute (SQoE-III, phone VMAF) and mean device energy per minute over the evaluation sessions.
- Sets: U = all energy-unaware points; A = U + energy-aware points; O = A + clairvoyant references.
- Frontier: Pareto frontier with linear interpolation between neighbouring points (reachable by time-sharing two configurations across sessions).
- Headroom:
  - H_aware = mean over points (q, E_U) on the U frontier of 1 − E_A(q)/E_U.
  - H_oracle: same with O instead of A.
- Secondary: Htop (highest-QoE point of U only); Hdata (data instead of energy); E2 decomposition H_watchonly, H_thronly.
- Uncertainty: session bootstrap (500 replicates for E1/E2/E3/E5, 300 for E4), 95% percentile intervals.

### E3: DRL vs rule-based methods
- Rule envelope = U ∪ A.
- Main metric G_drl: mean saving on the rule frontier after adding DRL to the rule envelope, i.e. H(rules → rules ∪ DRL); G_drl ≥ 0.
- Auxiliary: S_drl (saving of the DRL frontier relative to the rule envelope, only at QoE the DRL reaches); coverage (fraction of rule-frontier points the DRL reaches); highest QoE of both.

### E4: codecs
| Metric | Meaning |
|---|---|
| av1_vs_h264 | AV1 only vs H.264 only |
| both_vs_bestfixed | both codecs selectable vs best fixed-codec envelope |
| both_vs_caprule | vs the rule "use AV1 only with hardware AV1 decoding" |
| greedy_vs_h264 | energy-unaware, quality-only codec choice vs H.264 only |
| data_* | corresponding data savings |

## 5. Decision rules written in advance

**C1 (little left on modern phones)**
- Condition: under meas, the 95% upper endpoints of H_aware are < 5% for both video forms.
- Met: write "with measured parameters, energy-aware methods can save at most under 5% more".
- Not met: report the numbers only, without this sentence.

**C2 (prior settings overestimate)**
- Report the ratio of H_aware under legacy_lte to that under meas (point estimates).
- Write "significantly larger" only if the 95% lower endpoint under legacy exceeds the 95% upper endpoint under meas.

**C3 (headroom map)**
- Report H_aware and H_oracle at all 75 grid points.
- "Practically meaningful headroom" means a 95% lower endpoint of H_aware ≥ 10%; list the regions that meet it.
- Report Spearman correlations of H with tail energy, β and c (descriptive).

**C4 (gap to the clairvoyant reference)**
- Report H_oracle − H_aware and H_aware/H_oracle at each anchor, and the E2 decomposition. No direction is preset.

**C5 (DRL)**, per video form × regime:
| Condition | Conclusion |
|---|---|
| 95% upper endpoint of G_drl < 2% | DRL does not exceed tuned rule-based methods |
| 95% lower endpoint of G_drl > 2% | DRL adds savings that rules lack |
| otherwise | inconclusive |

**C6 (codecs)**
| Condition | Conclusion |
|---|---|
| with dav1d or the system decoder, on some network set, the 95% lower endpoint of both_vs_caprule ≥ 5% | decoder-energy-aware codec selection is worthwhile |
| otherwise | report numbers only |

Also report whether greedy_vs_h264 is negative without AV1 hardware decoding, i.e. whether quality-only codec switching costs energy.

**C7 (robustness)**
- Recompute H_aware under meas for each E5 variant: 2-s segments, standard VMAF, 5G traces, old traces, watch times sampled from retention curves.
- Write "robust" only if C1 holds in every variant; otherwise state where it fails.
- For long-form video, also report by the 9 genres (descriptive).

## 6. Scale
- E1/E2: 79 energy models × 246 feed configurations × 378 sessions, plus 79 × 187 long-form configurations × 252 sessions.
- E3: 2 forms × 4 regimes × 4 λ {0.02, 0.1, 0.4, 1.6} × 3 seeds = 96 DQNs, 5000 episodes each; network and hyper-parameters as in the earlier SPARE study's DQN.
- E4: 12 device/radio settings (11 after removing duplicates) × 3 codec modes × 3 network sets × 2 forms.
- The main evaluation set is run once. The development set is used only for local debugging.

## Revision log
- 2026-11-11 21:50: matplotlib was not installed on CPU2; the analysis script failed while plotting, and the pipeline stopped at the analysis of E1E2_short.
  - Fix: save the statistics tables first and make plotting optional; a failed analysis step no longer stops later simulations.
  - Impact: no simulation, metric or decision rule is affected. Completed simulation results were kept unchanged; only the analysis was re-run.
