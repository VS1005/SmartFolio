# Latent Factor Experiments (Sparse Autoencoder)

This folder is a self-contained playground for factorizing observations into sparse latent codes and measuring how well those codes explain the PPO policy logits.

## Workflow

1) **Collect traces**: run a small rollout to dump observations, PPO logits/actions, and rewards to a compressed NPZ.
2) **Train AE**: fit a sparse autoencoder on flattened observations; monitor reconstruction + sparsity losses.
3) **Align factors**: fit a linear ridge head from latent codes to PPO logits to quantify explainability (R² per action dim).

All artifacts (data/checkpoints/plots) are kept under this folder to avoid touching the main training/inference code.

## Usage (CLI)

Collect traces (no existing files touched):

```bash
python experiments/latent_factors/collect_traces.py \
  --model-path checkpoints/ppo_hgat.zip \
  --market hs300 --horizon 1 --relation-type hy \
  --test-start-date 2020-01-01 --test-end-date 2020-06-30 \
  --device cuda:0 \
  --output-name hs300_trace \
  --save-attention  # optional, larger files
```

Train sparse AE on the saved traces:

```bash
python experiments/latent_factors/train_ae.py \
  --data-path experiments/latent_factors/data/hs300_trace.npz \
  --latent-dim 16 --epochs 20 --sparsity-weight 1e-3 \
  --drop-prev-weights \             # optional: zero prev_weights block
  --prev-scale 0.1 --adj-scale 1.0 --ts-scale 1.0 \  # optional block scaling if not dropping
  --device cuda:0
```

Align factors to PPO logits with a linear ridge head:

```bash
python experiments/latent_factors/align_factors.py \
  --data-path experiments/latent_factors/data/hs300_trace.npz \
  --checkpoint experiments/latent_factors/checkpoints/sparse_ae.pt \
  --ridge-lambda 1e-3
```

Interpret and inspect factors:

```bash
python experiments/latent_factors/analyze_alignment.py \
  --outputs experiments/latent_factors/analysis/alignment_outputs.npz \
  --metrics experiments/latent_factors/analysis/alignment_metrics.json \
  --traces experiments/latent_factors/data/hs300_trace.npz \
  --tickers tickers.csv \
  --top-k 5 --top-stocks 5
```
