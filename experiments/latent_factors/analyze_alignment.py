#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import pearsonr
from preproc import slices_from_meta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize factor-to-logit alignment results")
    parser.add_argument(
        "--outputs",
        required=True,
        help="Path to alignment_outputs.npz produced by align_factors.py",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Optional path to alignment_metrics.json (if omitted, will look next to outputs)",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Optional traces NPZ (from collect_traces.py) to compute factor correlations with stocks/features",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Optional tickers CSV to map indices when traces are provided",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of strongest factors per action to report")
    parser.add_argument("--top-stocks", type=int, default=5, help="Number of stocks per factor to show (needs traces)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs_path = Path(args.outputs).expanduser()
    metrics_path = (
        Path(args.metrics).expanduser()
        if args.metrics is not None
        else outputs_path.with_name("alignment_metrics.json")
    )

    data = np.load(outputs_path)
    weights = data["weights"]  # shape (latent_dim + 1, num_actions); last row is bias
    latents = data["latents"]
    logits = data["logits"]
    preds = data["preds"]

    bias = weights[-1]  # shape (num_actions,)
    coef = weights[:-1]  # shape (latent_dim, num_actions)
    latent_dim, num_actions = coef.shape

    # Recompute R² and MSE for sanity
    ss_res = np.sum((logits - preds) ** 2, axis=0)
    ss_tot = np.sum((logits - np.mean(logits, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    mse = np.mean((logits - preds) ** 2, axis=0)

    print(f"Loaded alignment outputs from {outputs_path}")
    print(f"Latent dim: {latent_dim}, Actions: {num_actions}, Samples: {latents.shape[0]}")
    print(f"R² mean: {r2.mean():.4f} | MSE mean: {mse.mean():.6f}")

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        print(f"Metrics file: {metrics_path}")
        print(json.dumps(metrics, indent=2))

    top_k = max(1, min(args.top_k, latent_dim))
    print("\nTop factors per action (coef, signed):")
    for j in range(num_actions):
        coefs = coef[:, j]
        order = np.argsort(np.abs(coefs))[::-1][:top_k]
        summary = [(int(k), float(coefs[k])) for k in order]
        print(f"Action {j:02d}: {summary} | bias={float(bias[j]):.4f} | R²={float(r2[j]):.4f}")

    # Factor activity stats
    var = latents.var(axis=0)
    mean_abs = np.mean(np.abs(latents), axis=0)
    order = np.argsort(var)[::-1]
    print("\nFactor activity (variance, mean_abs):")
    for idx in order[: top_k * 2]:
        print(f"Factor {int(idx):02d}: var={float(var[idx]):.6f}, mean|z|={float(mean_abs[idx]):.6f}")

    # Optional: factor meaning via correlations to stocks/features using traces
    if args.traces:
        traces_path = Path(args.traces).expanduser()
        traces = np.load(traces_path, allow_pickle=True)
        meta = traces["meta"].item() if "meta" in traces.files else {}
        if not meta:
            meta_json = traces_path.with_suffix(".json")
            if meta_json.exists():
                with meta_json.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
        try:
            adj_slice, ts_slice, prev_slice = slices_from_meta(meta)
        except Exception as exc:
            print(f"\n[WARN] Unable to derive slices from meta, skipping per-stock correlations: {exc}")
            return

        obs = traces["obs"]
        tickers = []
        if args.tickers:
            import csv

            with Path(args.tickers).expanduser().open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames and "ticker" in reader.fieldnames:
                    tickers = [(row.get("ticker") or "").strip() for row in reader if (row.get("ticker") or "").strip()]

        prev_block = obs[:, prev_slice]  # shape (n, num_stocks)
        ts_block = obs[:, ts_slice]      # shape (n, num_stocks*lookback*feat)
        S = int(meta.get("num_stocks", prev_block.shape[1]))
        L = int(meta.get("lookback", 1))
        F = int(meta.get("input_dim", 1))
        ts = ts_block.reshape(obs.shape[0], S, L, F)
        # per-factor stock correlations using prev_weights as a proxy
        top_stocks = max(1, min(args.top_stocks, S))
        print("\nTop stocks per factor (corr with prev_weights):")
        for f_idx in range(latents.shape[1]):
            corrs = []
            for s in range(S):
                try:
                    corrs.append(pearsonr(latents[:, f_idx], prev_block[:, s])[0])
                except Exception:
                    corrs.append(0.0)
            order = np.argsort(np.abs(corrs))[::-1][:top_stocks]
            labels = []
            for s in order:
                label = tickers[s] if tickers and s < len(tickers) else f"Stock_{s}"
                labels.append((int(s), label, float(corrs[s])))
            print(f"Factor {f_idx:02d}: {labels}")

        # feature correlations (mean over stocks/time)
        feat_mean = ts.mean(axis=(1, 2))
        print("\nTop features per factor (corr with mean feature across stocks/time):")
        for f_idx in range(latents.shape[1]):
            corrs = []
            for f in range(F):
                try:
                    corrs.append(pearsonr(latents[:, f_idx], feat_mean[:, f])[0])
                except Exception:
                    corrs.append(0.0)
            order = np.argsort(np.abs(corrs))[::-1][:3]
            summary = [(int(f), float(corrs[f])) for f in order]
            print(f"Factor {f_idx:02d}: {summary}")


if __name__ == "__main__":
    main()
