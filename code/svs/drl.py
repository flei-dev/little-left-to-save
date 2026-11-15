"""Learned baseline: energy-aware deep RL preloader (dueling double DQN with prioritized n-step replay).

The agent sees the current video and the next WINDOW-1 videos (buffer, watch probability, expected watch time,
time until needed, size and VMAF of every rung of their next chunk), the throughput history and the radio
state, and picks one of WINDOW x 6 (video, rung) downloads or a 1-s sleep.  Reward per step:
   delta QoE (SQoE-III, phone VMAF) - lam * delta device energy [J]
so lam traces a QoE-energy front exactly like the lambda of the rule-based families (GreenABR / GreenABR+ idea).
It reuses the NumPy DQN in tmm_abr.

  train : python3 -m svs.drl train --data svs_data.pkl --traces <train trace dir> --radio lte --lam 0.2 --seed 0 --out drl_runs
  eval  : handled by cstudy.run --drl_runs <run dirs>, which loads <run dir>/weights.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .model import BITRATES, DEVICES, N_RUNGS, RADIOS, WINDOW
from .policies import Policy, feasible
from .sim import Session, Trace

SLEEP_S = 1.0
N_ACT = WINDOW * N_RUNGS + 1
STATE_DIM = 11 + WINDOW * (6 + 2 * N_RUNGS)


@dataclass
class DQNConfig:
    radio: str = "lte"
    device: str = "modern"
    lam: float = 0.2
    seed: int = 0
    regime: str = ""             # paper C (E3): regime label; "" keeps the old run ids
    form: str = "short"          # "short" (feed) or "long" (one long-form video with a buffer cap)
    buf_cap: float = math.inf
    episodes: int = 3000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 100_000
    n_step: int = 3
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 300_000
    target_update_steps: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay: float = 0.998
    warmup_episodes: int = 30
    learning_frequency: int = 4
    reward_scale: float = 0.1

    @property
    def run_id(self):
        if self.regime:
            return f"{self.form}_{self.regime}_lam{self.lam:g}_s{self.seed}"
        return f"{self.radio}_{self.device}_lam{self.lam:g}_s{self.seed}"


def featurize(S) -> tuple[np.ndarray, np.ndarray]:
    """State vector and the mask of valid actions."""
    rad = S.radio_belief
    x = np.zeros(STATE_DIM, np.float32)
    mask = np.zeros(N_ACT, bool)
    mask[-1] = True
    thr = S.thr[-5:]
    x[0] = S.thr_hm() / 10.0
    for i, v in enumerate(thr[::-1]):
        x[1 + i] = v / 10.0
    since = S.since_last()
    x[6] = min(since / rad.t_tail, 2.0) if math.isfinite(since) and rad.t_tail > 0 else 2.0
    x[7] = 1.0 if since <= rad.t_tail else 0.0
    c = S.C[S.cur]
    x[8] = S.pos / c.dur
    x[9] = min(S.exp_remaining(S.cur) / 30.0, 3.0)
    x[10] = 1.0 if S.stalling else 0.0
    o = 11
    for slot in range(WINDOW):
        j = S.cur + slot
        if j < S.nv:
            k = S.next_k(j)
            x[o] = 1.0
            x[o + 1] = min(S.buf_ahead(j) / 20.0, 3.0)
            x[o + 2] = len(S.dl[j]) / S.C[j].n
            if k is not None:
                x[o + 3] = S.p_watch(j, k)
                x[o + 4] = S.exp_watch(j, k) / S.chunk_s
                x[o + 5] = min(S.time_to_need(j, k) / 30.0, 3.0)
                x[o + 6:o + 6 + N_RUNGS] = S.C[j].sizes[k] / 1e6
                x[o + 6 + N_RUNGS:o + 6 + 2 * N_RUNGS] = S.C[j].q[k] / 100.0
                full = j == S.cur and S.buf_ahead(j) + S.chunk_s > S.buf_cap + 1e-9
                if not full:
                    for r in feasible(S, j, k):
                        mask[slot * N_RUNGS + r] = True
        o += 6 + 2 * N_RUNGS
    if S.stalling and mask[:-1].any():       # never idle while playback is blocked
        mask[-1] = False
    return x, mask


def decode(S, a):
    if a == N_ACT - 1:
        return ("sleep", SLEEP_S)
    slot, r = divmod(a, N_RUNGS)
    j = S.cur + slot
    return ("dl", j, S.next_k(j), r)


class DRLPolicy(Policy):
    """Greedy policy of a trained agent (or the exploring policy during training, via `trainer`)."""

    def __init__(self, qnet, lam: float, name: str = "DRL", trainer=None):
        self.q, self.lam, self.trainer = qnet, lam, trainer
        self.name = f"{name}(lam={lam:g})"

    def reset(self, S):
        self.prev = None

    def act(self, S):
        x, mask = featurize(S)
        if self.trainer is not None:
            a = self.trainer.step(S, x, mask)
        else:
            q = self.q(x[None, :])[0]
            q[~mask] = -np.inf
            a = int(np.argmax(q))
        return decode(S, a)


class Trainer:
    def __init__(self, agent, cfg: DQNConfig):
        self.agent, self.cfg = agent, cfg
        self.prev = None
        self.steps = 0
        self.ep_reward = 0.0

    def begin(self, S):
        self.prev = None
        self.snap = S.snapshot()
        self.ep_reward = 0.0

    def _reward(self, S, final=False):
        q, e = S.snapshot()
        if final:     # count the rest of the last tail, as the evaluation does
            from .model import radio_energy
            e += radio_energy(S.transfers, S.t, S.radio) - radio_energy(S.transfers, S.t, S.radio, full_tail=False)
        r = (q - self.snap[0]) - self.cfg.lam * (e - self.snap[1])
        self.snap = (q, e)
        return r * self.cfg.reward_scale

    def step(self, S, x, mask):
        ag = self.agent
        if self.prev is not None:
            r = self._reward(S)
            ag.remember(self.prev[0], self.prev[1], r, x, False)
            self.ep_reward += r
            self.steps += 1
            if self.ep_learn and self.steps % self.cfg.learning_frequency == 0:
                ag.learn()
        if ag.rng.random() < ag.epsilon:
            a = int(ag.rng.choice(np.flatnonzero(mask)))
        else:
            q = ag.q(x[None, :])[0]
            q[~mask] = -np.inf
            a = int(np.argmax(q))
        self.prev = (x, a)
        return a

    def end(self, S):
        if self.prev is not None:
            r = self._reward(S, final=True)
            self.agent.remember(self.prev[0], self.prev[1], r, np.zeros(STATE_DIM, np.float32), True)
            self.ep_reward += r
        self.agent.end_episode()


def load_traces(d):
    files = sorted(p for p in Path(d).iterdir() if p.is_file() and not p.name.startswith("."))
    return [np.loadtxt(f) for f in files]


def train(cfg: DQNConfig, data: str, traces: str, out: str, log_every: int = 50):
    from tmm_abr.agent import Agent
    run = Path(out) / cfg.run_id
    if (run / "done.json").exists():
        print("already done", run)
        return
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg)))
    d = pickle.load(open(data, "rb"))
    if "splits" in d:                                           # long-form: windows of the training videos
        clips, sessions = d["clips"], [d["sessions"][i] for i in d["splits"]["dev"]]
    else:                                                       # feed: even sessions (never evaluated)
        clips, sessions = d["clips"], d["sessions"][0::2]
    trs = load_traces(traces)
    rng = np.random.default_rng(cfg.seed)
    agent = Agent(cfg, STATE_DIM, N_ACT, rng)
    tr = Trainer(agent, cfg)
    pol = DRLPolicy(agent.q, cfg.lam, trainer=tr)
    t0 = time.time()
    with (run / "train_log.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "reward", "qoe_min", "E_min", "epsilon", "minutes"])
        for ep in range(cfg.episodes):
            a = trs[int(rng.integers(len(trs)))]
            items = sessions[int(rng.integers(len(sessions)))][1]
            S = Session(clips, items, Trace(a[:, 0], a[:, 1], float(rng.uniform(0, a[-1, 0]))),
                        RADIOS[cfg.radio], DEVICES[cfg.device], buf_cap=cfg.buf_cap)
            tr.ep_learn = ep >= cfg.warmup_episodes
            tr.begin(S)
            pol.reset(S)
            S.done = False
            while not S.done and S.t < 3600:
                act = pol.act(S)
                if act[0] == "dl":
                    S.fetch(act[1], act[3])
                else:
                    S.advance(S.t + act[1])
            tr.end(S)
            if ep >= cfg.warmup_episodes:
                agent.decay_epsilon()
            if ep % log_every == 0 or ep == cfg.episodes - 1:
                m = S.metrics()
                w.writerow([ep, round(tr.ep_reward, 3), round(m["qoe_min"], 2), round(m["E_min"], 1),
                            round(agent.epsilon, 3), round((time.time() - t0) / 60, 1)])
                fh.flush()
    np.savez(run / "weights.npz", **agent.q.get_weights())
    (run / "done.json").write_text(json.dumps({"minutes": (time.time() - t0) / 60}))
    print("done", run, f"{(time.time() - t0) / 60:.0f} min")


def load_policy(run_dir, lam):
    from tmm_abr.nn import DuelingQNet
    q = DuelingQNet(STATE_DIM, N_ACT, np.random.default_rng(0))
    wts = dict(np.load(Path(run_dir) / "weights.npz"))
    q.set_weights(wts)
    pol = DRLPolicy(q, lam)
    pol.cap = json.loads((Path(run_dir) / "config.json").read_text()).get("buf_cap", math.inf)
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--radio", default="lte")
    ap.add_argument("--device", default="modern")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--out", default="drl_runs")
    a = ap.parse_args()
    cfg = DQNConfig(radio=a.radio, device=a.device, lam=a.lam, seed=a.seed, episodes=a.episodes)
    train(cfg, a.data, a.traces, a.out)


if __name__ == "__main__":
    main()
