"""Run the spike: generate pairs, extract attention, compute H1 features per head."""
from __future__ import annotations

import os
import pandas as pd

from src.data_gen import build_pairs
from src.attn_extract import load_model, get_attentions, is_correct
from src.topology import h1_features

THRESHOLDS = (0.3, 0.5, 0.7)
TOP_K = 8


def main(n_per_family: int = 30, seed: int = 0, out_path: str = "results/spike.parquet"):
    os.makedirs("results", exist_ok=True)
    items = build_pairs(n_per_family=n_per_family, seed=seed)
    model, tok, device = load_model()

    rows = []
    for idx, it in enumerate(items):
        ok, ans = is_correct(model, tok, device, it.prompt, it.gold)
        atts = get_attentions(model, tok, device, it.prompt)  # [L, H, n, n]
        n_layers, n_heads, n, _ = atts.shape
        for layer in range(n_layers):
            for head in range(n_heads):
                feats = h1_features(
                    atts[layer, head], thresholds=THRESHOLDS, top_k=TOP_K
                )
                rows.append({
                    "item_id": it.id, "pair_id": it.pair_id, "family": it.family,
                    "hop": it.hop, "correct": ok, "seq_len": n,
                    "layer": layer, "head": head, **feats,
                })
        print(f"[{idx+1}/{len(items)}] {it.id} correct={ok} ans={ans!r}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"Overall task accuracy: {df.groupby('item_id')['correct'].first().mean():.2%}")
    return df


if __name__ == "__main__":
    main()
