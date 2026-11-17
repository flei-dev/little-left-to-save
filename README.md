# Little Left to Save? — code, data and pre-registration

Code, data, pre-registration and result summaries for the paper

> **Little Left to Save? Measurement-Calibrated Energy Headroom of Adaptive Video Streaming on Modern Smartphones**

The study measures radio power on two smartphones, calibrates a common simulator for long-form video and short-video feeds, and compares tuned energy-unaware players, energy-aware players and a clairvoyant reference at matched quality of experience (QoE) across a 75-point energy-model grid.

## Repository layout

| Folder | Contents |
|---|---|
| `prereg/` | Pre-registration. `PREREG_C.md` is the governing text (Chinese); `PREREG_C_frozen_2026-11-11.md` is the file as frozen before the server runs; `PREREG_C_en.md` is an English translation. |
| `code/svs/` | Simulator: radio and device energy model (`model.py`), feed sessions (`sim.py`), feed policies (`policies.py`), long-form policies (`longform.py`), policy grids (`grids.py`), DQN baseline (`drl.py`). |
| `code/cstudy/` | Study driver: energy-model grid and anchors (`models.py`), stage runner (`run.py`), headroom analysis and bootstrap (`analyze.py`), AV1 data merge (`codec_data.py`), pipeline (`main.py`). |
| `code/tmm_abr/` | NumPy DQN (dueling + double DQN, prioritized 3-step replay) used by `svs/drl.py`. |
| `code/encode_vmaf.py`, `code/run_all.py` | AV1 / H.264 encoding and per-segment VMAF (standard and phone models). |
| `data/` | Content library (46 videos × 6 rungs; 4-s and 2-s segments; H.264 and AV1), simulator inputs (`*.pkl`), and `videos.csv` (title, genre, source URL of each video). |
| `traces/` | Network traces cut into 300-s windows: `lte/`, `5g/`, `old/`, each with `train/` and `test/`. |
| `results/` | Summary tables of every stage (`<stage>/summary/*.csv`), configuration means, evaluation pairs, run arguments and logs. |
| `phone/` | Termux measurement scripts, raw battery logs of both phones, and their analysis. |
| `figures/` | Scripts that draw every figure of the paper and its supplement. |

## Stages and where the paper's numbers come from

| Stage (`results/`) | Paper | Key file → field |
|---|---|---|
| `E1E2_short`, `E1E2_long` | Fig. 2, Fig. 6, Sec. V-A | `summary/headroom.csv` → `H_aware`, `H_oracle`, `*_lo`, `*_hi` (rows `meas`, `legacy_lte`, `modern_lte`, `modern_5g`, and 75 grid rows `c*_T*_b*`) |
| `energy_split_components.csv` | Fig. 7, Sec. V-B | `H_total`, `Eradio_share`, `Ec_share`, `Estall_share`, `U_*` (energy-unaware components) |
| `E2_decomp_short` | Sec. V-B | `summary/headroom.csv` → `H_watchonly`, `H_thronly`, `H_oracle` |
| `E3_<form>_<regime>` | Sec. V-C | `summary/drl.csv` → `G_drl`, `S_drl_vs_rules`, `coverage`, `qtop_*` |
| `E4_<form>_<network>`, `E4_data` | Fig. 8, Sec. V-D | `summary/codec.csv` → `av1_vs_h264`, `greedy_vs_h264`, `both_vs_caprule`, `data_av1_vs_h264` (models `*_prop_lte_meas`; flat-scaling sensitivity `*_flat_lte_meas`) |
| `E5_<form>_<variant>`, `E1E2_long/summary/genre.csv` | Fig. 9, Sec. V-E | `summary/headroom.csv` → `H_aware` |
| `phone/data/` | Fig. 5, Sec. III-B/C | `tail_energy_table.json`; `python3 phone/analyze_decode.py` |

Intervals are 95% session-bootstrap percentile intervals (500 replicates; 300 for E4).

## Requirements

Python 3.10 or newer with NumPy and pandas; Matplotlib for figures. The results were produced on Linux servers with the system `python3`. Install with

```
pip install -r requirements.txt
```

E4 re-encoding additionally needs FFmpeg with libx264, libsvtav1 and libvmaf, and the source videos listed in `data/videos.csv`.

## Reproducing

Run every command from the repository root.

**Figures from the shipped results** (a few seconds; 300-dpi PNGs identical to the figures in the paper):

```
mkdir -p figures/out
PYTHONPATH=code python3 figures/make_figs.py results figures/out
PYTHONPATH=code python3 figures/make_figs2.py results results figures/out
python3 figures/make_fig_phone.py figures/out
python3 figures/fig2.py figures/out
python3 figures/fig_problem.py . figures/out
python3 figures/fig_energy_model.py figures/out
python3 figures/decomp_components.py . results/energy_split_components.csv
python3 figures/fig_energy_split.py results/energy_split_components.csv figures/out
```

`decomp_components.py` splits the headroom of each anchor exactly into radio, base-playback, per-megabyte decoding and stall components, using the same Pareto fronts and interpolation as the headroom metric (Proposition 1 of the paper).

`results/<stage>/summary/cfg_means.csv` holds the mean QoE, energy and data of every configuration under every energy model, from which the frontiers can be redrawn. The per-session raw outputs needed for the bootstrap are not included; a full re-run (below) regenerates them, and `python3 -m cstudy.analyze headroom results/<stage> --boot 500` then recomputes the summary tables.

**Phone measurements:**

```
python3 phone/analyze_decode.py                      # decoder power, Sec. III-C
python3 phone/analyze_phone.py phone/data/redmi_5g --out /tmp/redmi_5g
```

`phone/svs_phone.py` (radio tail and transfer power) and `phone/decode_test.py` (real-time decoding power) run on an Android phone in Termux with Termux:API.

**Full re-run of the simulations.** The pipeline skips every stage that has a `DONE` marker, so move the shipped `results/` folder away first. Then:

```
bash run.sh cpu2      # E1, E2, E3, E5 (feed and long-form)
bash run.sh cpu1      # AV1 encoding, then E4
bash run.sh status
```

The original runs used 31 worker processes per server and took about two days. E4 needs the AV1 data files; to build them from the shipped AV1 table instead of re-encoding:

```
PYTHONPATH=code python3 -m cstudy.codec_data --h264 data/content_library_x264_2s.csv \
    --av1_csv data/content_library_av1_2s.csv --short data/svs_data_2s.pkl --out_dir work/data_codec
```

E3 trains 96 DQNs from scratch (5000 episodes each); the trained weights are not included, only the training logs in `results/E3_train/runs/`.

## Notes

- `phone/analyze_decode.py` skips the first 3 s and last 1 s of each 60-s decoding interval. It gives an extra power over hardware H.264 of 0.170 W (sports) and 0.058 W (talk) for dav1d, and 0.884 W and 0.495 W for the system decoder; the energy model uses the means of the per-clip values rounded to two decimals (0.115 W and 0.69 W).
- The figure scripts set fonts and sizes for IEEE two-column layout (300 dpi).

## Data sources

The repository redistributes derived data. Please also cite the original sources.

| Data | Source | License |
|---|---|---|
| Viewing sessions and retention curves (`data/svs_data*.pkl`) | KuaiRand-Pure (Gao et al., CIKM 2022), Zenodo record 10439422 | CC BY 4.0 |
| LTE traces (`traces/lte`) | Raca et al., "Beyond throughput: a 4G LTE dataset with channel and context metrics", MMSys 2018, Zenodo record 1219679 | CC BY 4.0 |
| 5G traces (`traces/5g`) | Raca et al., "Beyond throughput, the next generation: a 5G dataset with channel and context metrics", MMSys 2020, github.com/uccmisl/5Gdataset | see the dataset's repository |
| Low-throughput traces (`traces/old`) | Mixture of 3G (HSDPA), FCC broadband and synthetic traces as distributed with Pensieve (Mao et al., SIGCOMM 2017) | see the original repositories |
| Content library (`data/content_library_*.csv`) | 43 Creative Commons (CC BY) YouTube videos and three Blender Foundation films, listed in `data/videos.csv`; only segment sizes and quality scores are distributed, not the videos | CC BY 4.0 for the derived tables |
| Phone measurements (`phone/data`) | Measured by the authors (Huawei MAO-AL00 and Redmi 2311DRK48C, China Mobile) | CC BY 4.0 |

## License

Code: MIT (see `LICENSE`). Derived data and measurements created for this study: CC BY 4.0; third-party data keep their original licenses (table above).

## Integrity

`MANIFEST.sha256` lists the SHA-256 hash of every file as stored in this repository (text files with LF line endings). Check with `sha256sum -c MANIFEST.sha256` on a Linux or macOS clone; Git for Windows may convert line endings on checkout.
