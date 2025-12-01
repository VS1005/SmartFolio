#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TraceDataset
from model import SparseAutoencoder
from preproc import PreprocessConfig, preprocess_obs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align latent factors to PPO logits with a linear head")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ")
    parser.add_argument("--checkpoint", required=True, help="Path to sparse_ae.pt")
    parser.add_argument("--output-dir", default="experiments/latent_factors/analysis", help="Where to store outputs")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge regularization strength")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    return parser.parse_args(argv)


def load_autoencoder(ckpt_path: Path, device: torch.device) -> Tuple[SparseAutoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    input_dim = ckpt.get("input_dim")
    latent_dim = config.get("latent_dim")
    hidden_dim = config.get("hidden_dim", 512)
    num_hidden = config.get("num_hidden", 2)
    if input_dim is None or latent_dim is None:
        raise ValueError("Checkpoint missing input_dim or latent_dim")
    model = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    preproc_cfg = PreprocessConfig.from_dict(ckpt.get("preprocess"))
    meta = ckpt.get("meta")
    return model, {"config": config, "preprocess": preproc_cfg, "meta": meta}


def fit_ridge(Z: np.ndarray, Y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge regression with bias term."""
    ones = np.ones((Z.shape[0], 1), dtype=Z.dtype)
    Zb = np.concatenate([Z, ones], axis=1)
    dim = Zb.shape[1]
    A = Zb.T @ Zb + lam * np.eye(dim, dtype=Z.dtype)
    B = Zb.T @ Y
    W = np.linalg.solve(A, B)
    preds = Zb @ W
    return preds, W


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return r2


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TraceDataset(args.data_path, reshape=False)
    if dataset.logits is None:
        raise ValueError("Traces file does not contain 'logits'; cannot perform alignment.")

    device = torch.device(args.device)
    model, meta_cfg = load_autoencoder(Path(args.checkpoint).expanduser(), device)
    preproc_cfg: PreprocessConfig = meta_cfg["preprocess"]
    ckpt_meta = meta_cfg.get("meta") or {}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latents = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            obs_raw = batch["obs_flat"].to(device)
            # Use metadata from checkpoint if present, else dataset meta
            meta_for_preproc = ckpt_meta if ckpt_meta else dataset.meta
            obs = preprocess_obs(obs_raw, meta_for_preproc, preproc_cfg)
            _, z = model(obs)
            latents.append(z.cpu().numpy())
            targets.append(batch["logits"].numpy())
    Z = np.concatenate(latents, axis=0)
    Y = np.concatenate(targets, axis=0)

    preds, weights = fit_ridge(Z, Y, args.ridge_lambda)
    r2 = r2_score(Y, preds)
    mse = np.mean((Y - preds) ** 2, axis=0)

    metrics = {
        "ridge_lambda": args.ridge_lambda,
        "r2_mean": float(np.mean(r2)),
        "r2_per_action": r2.tolist(),
        "mse_mean": float(np.mean(mse)),
        "mse_per_action": mse.tolist(),
        "num_samples": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
    }
    metrics_path = output_dir / "alignment_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved alignment metrics to {metrics_path}")

    np.savez_compressed(
        output_dir / "alignment_outputs.npz",
        latents=Z,
        logits=Y,
        preds=preds,
        weights=weights,
    )
    print(f"Saved alignment outputs to {output_dir / 'alignment_outputs.npz'}")


if __name__ == "__main__":
    main()
