#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import TraceDataset
from model import SparseAutoencoder, sparse_ae_loss
from preproc import PreprocessConfig, preprocess_obs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse autoencoder on PPO traces")
    parser.add_argument("--data-path", required=True, help="Path to traces NPZ produced by collect_traces.py")
    parser.add_argument("--output-dir", default="experiments/latent_factors/checkpoints", help="Where to store checkpoints")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer width")
    parser.add_argument("--num-hidden", type=int, default=2, help="Number of hidden layers in encoder/decoder")
    parser.add_argument("--sparsity-weight", type=float, default=1e-3, help="L1 penalty weight on latent activations")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--drop-prev-weights", action="store_true", help="Zero out prev_weights block during training")
    parser.add_argument("--adj-scale", type=float, default=1.0, help="Scale factor for adjacency block")
    parser.add_argument("--ts-scale", type=float, default=1.0, help="Scale factor for ts_features block")
    parser.add_argument(
        "--prev-scale",
        type=float,
        default=0.1,
        help="Scale factor for prev_weights block (ignored if --drop-prev-weights)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TraceDataset(args.data_path, reshape=False)
    input_dim = dataset.obs.shape[1]
    preproc_cfg = PreprocessConfig(
        drop_prev=args.drop_prev_weights,
        adj_scale=args.adj_scale,
        ts_scale=args.ts_scale,
        prev_scale=args.prev_scale,
    )

    total_len = len(dataset)
    val_size = max(1, int(total_len * args.val_split))
    train_size = total_len - val_size
    if train_size <= 0:
        train_size = 1
        val_size = max(0, total_len - train_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = SparseAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_sparsity": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            obs = preprocess_obs(batch["obs_flat"].to(device), dataset.meta, preproc_cfg)
            optimizer.zero_grad()
            recon, latent = model(obs)
            losses = sparse_ae_loss(recon, obs, latent, args.sparsity_weight)
            losses["loss"].backward()
            optimizer.step()
            train_loss += float(losses["loss"].item()) * obs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_sparsity = 0.0
        if len(val_loader.dataset) > 0:
            with torch.no_grad():
                for batch in val_loader:
                    obs = preprocess_obs(batch["obs_flat"].to(device), dataset.meta, preproc_cfg)
                    recon, latent = model(obs)
                    losses = sparse_ae_loss(recon, obs, latent, args.sparsity_weight)
                    val_loss += float(losses["loss"].item()) * obs.size(0)
                    val_mse += float(losses["mse"].item()) * obs.size(0)
                    val_sparsity += float(losses["sparsity"].item()) * obs.size(0)
            val_loss /= len(val_loader.dataset)
            val_mse /= len(val_loader.dataset)
            val_sparsity /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_sparsity"].append(val_sparsity)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} mse={val_mse:.6f} sparsity={val_sparsity:.6f}"
        )

    ckpt_path = output_dir / "sparse_ae.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "input_dim": input_dim,
            "preprocess": preproc_cfg.to_dict(),
            "meta": dataset.meta,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Export latent codes for validation split
    latents: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            obs = preprocess_obs(batch["obs_flat"].to(device), dataset.meta, preproc_cfg)
            _, latent = model(obs)
            latents.append(latent.cpu().numpy())
            if "logits" in batch:
                targets.append(batch["logits"].cpu().numpy())
    if latents:
        latents_arr = np.concatenate(latents, axis=0)
        export = {"latents": latents_arr}
        if targets:
            export["logits"] = np.concatenate(targets, axis=0)
        np.savez_compressed(output_dir / "val_latents.npz", **export)
        print(f"Saved validation latents to {output_dir / 'val_latents.npz'}")


if __name__ == "__main__":
    main()
