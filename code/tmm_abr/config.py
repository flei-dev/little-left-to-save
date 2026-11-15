"""Run configuration, state variants and train/test splits for the TMM content-generalisation study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

# State variants (all share the 7 basic inputs: last bitrate, buffer, throughput, delay,
# chunks left, last energy, last quality).
#   esmrb     : basic state only
#   lookraw   : + next chunk's raw VMAF and size of every rung (Comyco-style lookahead, 12 inputs)
#   marginal  : + video-agnostic marginal features of the next chunk (proposed, 17 inputs):
#               download-time ratio of every rung (6), VMAF gain of each step up (5),
#               VMAF gain per extra MB of each step up (5), headroom to the top rung (1)
# Augmentation ("aug"): per training episode, the quality-rate curve and the sizes of the
# sampled video are randomly perturbed (see env.Perturbation).
VARIANTS = ("esmrb", "lookraw", "marginal")


@dataclass
class RunConfig:
    variant: str = "marginal"
    aug: bool = False
    split: str = "video"          # "video": 5 folds of held-out videos; "genre": leave one genre out;
                                  # "blender": train on the 3 Blender films only, test on the other 43
    fold: int = 0
    n_train: int = 0              # >0: train on a random subset of n training videos of the fold
    quality: str = "phone"        # "phone": phone VMAF model; "std": default VMAF model
    train_device: str = "galaxy"
    seed: int = 0
    episodes: int = 30_000
    # DQN hyper-parameters
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 50_000
    warmup_episodes: int = 200
    learning_frequency: int = 4
    target_update_steps: int = 1_000
    n_step: int = 3
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9999
    train_chunks: int = 44
    # reward weights
    quality_weight: float = 1.5
    rebuffer_penalty: float = 1.0
    smoothness_penalty: float = 0.5
    energy_weight: float = 0.5
    # augmentation ranges
    aug_gamma: tuple = (0.6, 1.6)     # VMAF' = 100 * (VMAF/100) ** g
    aug_size: tuple = (0.7, 1.4)      # all sizes of the episode scaled by s
    calib_episodes: int = 300
    tag: str = ""

    @property
    def run_id(self) -> str:
        a = "aug" if self.aug else "noaug"
        t = f"_{self.tag}" if self.tag else ""
        n = f"_n{self.n_train}" if self.n_train else ""
        q = "" if self.quality == "phone" else f"_q{self.quality}"
        w = "" if self.energy_weight == 0.5 else f"_w{self.energy_weight:g}"
        return f"{self.variant}_{a}_{self.split}{self.fold}{n}{q}{w}_{self.train_device}_s{self.seed}{t}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        d = json.loads(s)
        for k in ("aug_gamma", "aug_size"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
