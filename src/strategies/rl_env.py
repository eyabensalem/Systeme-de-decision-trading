from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats(df: pd.DataFrame, feature_cols: list[str]) -> NormStats:
    X = df[feature_cols].astype(float).values
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return NormStats(mean=mean, std=std)


class TradingEnv(gym.Env):
    """
    Discrete actions: 0=SHORT(-1), 1=FLAT(0), 2=LONG(+1)
    Reward = pos_{t} * ret_{t+1} - cost * |pos_t - pos_{t-1}|
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        price_col: str = "close_15m",
        transaction_cost: float = 0.00005,
        norm: NormStats | None = None,
    ):
        super().__init__()
        self.df = df.sort_values("timestamp").reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.tc = float(transaction_cost)

        # precompute returns (log returns)
        self.ret = np.log(self.df[price_col].astype(float).values)
        self.ret = np.diff(self.ret, prepend=np.nan)  # ret[t] = log(p[t]) - log(p[t-1])

        X = self.df[feature_cols].astype(float).values
        if norm is None:
            norm = compute_norm_stats(self.df, feature_cols)
        self.norm = norm
        self.X = (X - self.norm.mean) / self.norm.std
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32
        )

        self.t = 0
        self.pos_prev = 0  # -1,0,1
        self.pos = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1  # start at 1 because ret[0] is nan
        self.pos_prev = 0
        self.pos = 0
        obs = self.X[self.t].astype(np.float32)
        info = {}
        return obs, info

    def step(self, action: int):
        # map action -> position
        if action == 0:
            self.pos = -1
        elif action == 1:
            self.pos = 0
        else:
            self.pos = 1

        # reward uses next return (t -> t+1) without lookahead: apply pos at t to ret at t+1
        # ret[t+1] exists if t+1 < len
        done = False
        truncated = False

        if self.t + 1 >= len(self.df):
            done = True
            obs = self.X[self.t].astype(np.float32)
            return obs, 0.0, done, truncated, {"pos": self.pos}

        r_next = float(self.ret[self.t + 1])
        cost = self.tc * abs(self.pos - self.pos_prev)
        reward = (self.pos * r_next) - cost

        self.pos_prev = self.pos
        self.t += 1

        obs = self.X[self.t].astype(np.float32)
        info = {"pos": self.pos, "cost": cost, "ret_next": r_next}
        return obs, float(reward), done, truncated, info
