from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npz(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    data = np.load(npz_path, allow_pickle=True)
    fields = {k: data[k] for k in data.files}

    meta_path = npz_path.with_suffix(".json")
    meta: Dict[str, object] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return fields, meta


def _split_observation(obs: np.ndarray, num_stocks: int, lookback: int, input_dim: int) -> Dict[str, np.ndarray]:
    """Split flattened observation into adjacency, ts_features, and prev_weights."""
    adj_size = num_stocks * num_stocks
    ts_size = num_stocks * lookback * input_dim
    prev_size = num_stocks

    expected = adj_size * 3 + ts_size + prev_size
    if obs.shape[-1] != expected:
        raise ValueError(f"Observation length {obs.shape[-1]} does not match expected {expected}")

    cursor = 0
    adj_ind = obs[..., cursor : cursor + adj_size]
    cursor += adj_size
    adj_pos = obs[..., cursor : cursor + adj_size]
    cursor += adj_size
    adj_neg = obs[..., cursor : cursor + adj_size]
    cursor += adj_size

    ts_flat = obs[..., cursor : cursor + ts_size]
    cursor += ts_size

    prev_weights = obs[..., cursor : cursor + prev_size]

    adj = np.stack(
        [
            adj_ind.reshape(num_stocks, num_stocks),
            adj_pos.reshape(num_stocks, num_stocks),
            adj_neg.reshape(num_stocks, num_stocks),
        ],
        axis=0,
    )
    ts_features = ts_flat.reshape(num_stocks, lookback, input_dim)
    return {"adj": adj, "ts_features": ts_features, "prev_weights": prev_weights}


class TraceDataset(Dataset):
    """Dataset for sparse AE training using saved traces."""

    def __init__(self, npz_path: str | Path, reshape: bool = True):
        npz_path = Path(npz_path).expanduser()
        fields, meta = _load_npz(npz_path)

        self.obs = fields.get("obs")
        if self.obs is None:
            raise ValueError(f"'obs' not found in {npz_path}")
        self.logits = fields.get("logits")
        self.actions = fields.get("actions")
        self.rewards = fields.get("rewards")
        self.dones = fields.get("dones")
        self.meta = meta
        self.reshape = reshape

        self.num_stocks = int(meta.get("num_stocks") or 0)
        self.lookback = int(meta.get("lookback") or 0)
        self.input_dim = int(meta.get("input_dim") or 0)

        if reshape and (self.num_stocks == 0 or self.lookback == 0 or self.input_dim == 0):
            raise ValueError("Missing num_stocks/lookback/input_dim in metadata; cannot reshape observations.")

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs_flat = self.obs[idx]
        sample: Dict[str, torch.Tensor] = {"obs_flat": torch.as_tensor(obs_flat, dtype=torch.float32)}

        if self.reshape:
            parts = _split_observation(obs_flat, self.num_stocks, self.lookback, self.input_dim)
            sample["adj"] = torch.as_tensor(parts["adj"], dtype=torch.float32)
            sample["ts_features"] = torch.as_tensor(parts["ts_features"], dtype=torch.float32)
            sample["prev_weights"] = torch.as_tensor(parts["prev_weights"], dtype=torch.float32)

        if self.logits is not None:
            sample["logits"] = torch.as_tensor(self.logits[idx], dtype=torch.float32)
        if self.actions is not None:
            sample["actions"] = torch.as_tensor(self.actions[idx], dtype=torch.float32)
        if self.rewards is not None:
            sample["reward"] = torch.as_tensor(self.rewards[idx], dtype=torch.float32)
        if self.dones is not None:
            sample["done"] = torch.as_tensor(self.dones[idx], dtype=torch.bool)

        return sample
