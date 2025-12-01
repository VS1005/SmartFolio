from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch


@dataclass
class PreprocessConfig:
    drop_prev: bool = False
    adj_scale: float = 1.0
    ts_scale: float = 1.0
    prev_scale: float = 0.1

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float | bool] | None) -> "PreprocessConfig":
        if data is None:
            return cls()
        return cls(
            drop_prev=bool(data.get("drop_prev", False)),
            adj_scale=float(data.get("adj_scale", 1.0)),
            ts_scale=float(data.get("ts_scale", 1.0)),
            prev_scale=float(data.get("prev_scale", 0.1)),
        )


def slices_from_meta(meta: Dict[str, object]) -> Tuple[slice, slice, slice]:
    num_stocks = int(meta.get("num_stocks"))
    lookback = int(meta.get("lookback"))
    input_dim = int(meta.get("input_dim"))
    adj_size = num_stocks * num_stocks
    ts_size = num_stocks * lookback * input_dim
    prev_size = num_stocks
    adj_slice = slice(0, adj_size * 3)
    ts_slice = slice(adj_slice.stop, adj_slice.stop + ts_size)
    prev_slice = slice(ts_slice.stop, ts_slice.stop + prev_size)
    return adj_slice, ts_slice, prev_slice


def preprocess_obs(obs: torch.Tensor, meta: Dict[str, object], cfg: PreprocessConfig) -> torch.Tensor:
    """Apply block-wise scaling and optional dropping of prev_weights."""
    adj_slice, ts_slice, prev_slice = slices_from_meta(meta)
    obs = obs.clone()
    if cfg.adj_scale != 1.0:
        obs[..., adj_slice] *= cfg.adj_scale
    if cfg.ts_scale != 1.0:
        obs[..., ts_slice] *= cfg.ts_scale
    if cfg.drop_prev:
        obs[..., prev_slice] = 0.0
    elif cfg.prev_scale != 1.0:
        obs[..., prev_slice] *= cfg.prev_scale
    return obs
