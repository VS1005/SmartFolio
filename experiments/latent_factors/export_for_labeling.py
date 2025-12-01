#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a small slice of data for factor labeling")
    parser.add_argument(
        "--traces",
        required=True,
        help="Path to traces NPZ (from collect_traces.py) containing obs/logits/actions",
    )
    parser.add_argument(
        "--alignment",
        required=True,
        help="Path to alignment_outputs.npz containing latents/logits/preds/weights",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional path to metadata JSON (defaults to traces .json sibling)",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Optional tickers CSV with 'ticker' column to map action indices",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of timesteps/samples to export (sequential from start)",
    )
    parser.add_argument(
        "--output",
        default="experiments/latent_factors/analysis/labeling_payload.npz",
        help="Where to write the export NPZ",
    )
    return parser.parse_args(argv)


def load_tickers(csv_path: Path | None) -> list[str]:
    if csv_path is None or not csv_path.exists():
        return []
    import csv

    tickers: list[str] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and "ticker" in reader.fieldnames:
            for row in reader:
                value = (row.get("ticker") or "").strip()
                if value:
                    tickers.append(value)
        else:
            fh.seek(0)
            raw = csv.reader(fh)
            for idx, row in enumerate(raw):
                if not row:
                    continue
                value = (row[0] or "").strip()
                if idx == 0 and value.lower() == "ticker":
                    continue
                if value:
                    tickers.append(value)
    return tickers


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    traces_path = Path(args.traces).expanduser()
    alignment_path = Path(args.alignment).expanduser()
    meta_path = Path(args.meta).expanduser() if args.meta else traces_path.with_suffix(".json")
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traces = np.load(traces_path, allow_pickle=True)
    alignment = np.load(alignment_path, allow_pickle=True)

    obs = traces["obs"]
    logits = traces["logits"] if "logits" in traces else None
    actions = traces["actions"] if "actions" in traces else None

    latents = alignment["latents"]
    aligned_logits = alignment["logits"]
    preds = alignment["preds"]
    weights = alignment["weights"]

    n = min(args.count, latents.shape[0], obs.shape[0])
    obs_slice = obs[:n]
    logits_slice = logits[:n] if logits is not None else None
    actions_slice = actions[:n] if actions is not None else None
    latents_slice = latents[:n]
    aligned_logits_slice = aligned_logits[:n]
    preds_slice = preds[:n]

    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as fh:
            try:
                meta = json.load(fh)
            except json.JSONDecodeError:
                meta = {}

    tickers = []
    if args.tickers:
        tickers = load_tickers(Path(args.tickers).expanduser())

    np.savez_compressed(
        output_path,
        obs=obs_slice,
        logits=logits_slice,
        actions=actions_slice,
        latents=latents_slice,
        aligned_logits=aligned_logits_slice,
        preds=preds_slice,
        weights=weights,
        meta=meta,
        tickers=np.asarray(tickers, dtype=object),
    )
    print(f"Exported {n} samples to {output_path}")
    print(f"Included meta keys: {list(meta.keys())}")
    if tickers:
        print(f"Included {len(tickers)} tickers")


if __name__ == "__main__":
    main()
