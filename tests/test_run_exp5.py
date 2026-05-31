import numpy as np
import pytest

import torch

from src.run_exp5 import (
    second_copy_loss, all_candidate_edges,
    _select_edges_by_magnitude, _select_edges_random,
)

slow = pytest.mark.slow


def test_second_copy_loss_scores_only_second_copy():
    # n = prefix(2) + S(3) + S(3) = 8; second-copy targets at t=5,6,7
    prefix_len, seq_len = 2, 3
    n = prefix_len + 2 * seq_len
    vocab = 5
    input_ids = torch.arange(n).remainder(vocab).unsqueeze(0)  # [1, n]
    # logits that put all mass on the gold token at every position -> loss ~ 0
    logits = torch.full((1, n, vocab), -10.0)
    for t in range(1, n):
        logits[0, t - 1, int(input_ids[0, t])] = 50.0
    loss = second_copy_loss(logits, input_ids, prefix_len, seq_len)
    assert loss < 1e-3


def test_all_candidate_edges_upper_triangular_positive():
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 0] = 0.5
    W[1, 2] = W[2, 1] = 0.2
    cand = all_candidate_edges(W)
    assert cand == {frozenset((0, 1)), frozenset((1, 2))}


def test_select_edges_by_magnitude_picks_strongest():
    W = np.zeros((3, 3))
    W[0, 1] = W[1, 0] = 0.9
    W[1, 2] = W[2, 1] = 0.1
    cand = {frozenset((0, 1)), frozenset((1, 2))}
    assert _select_edges_by_magnitude(W, cand, 1) == {frozenset((0, 1))}


def test_select_edges_random_is_seeded_and_right_size():
    cand = {frozenset((0, 1)), frozenset((1, 2)), frozenset((2, 3))}
    a = _select_edges_random(cand, 2, seed=42)
    b = _select_edges_random(cand, 2, seed=42)
    assert a == b and len(a) == 2 and a <= cand


@slow
def test_run_exp5_smoke(tmp_path):
    import pandas as pd
    cyc = tmp_path / "cycles.parquet"
    abl = tmp_path / "ablation.parquet"
    torch.set_num_threads(2)
    cycles_df, ablation_df = __import__("src.run_exp5", fromlist=["run"]).run(
        out_cycles=str(cyc), out_ablation=str(abl),
        model_name="gpt2", seq_len=8, prefix_len=3, n_seqs=2,
        top_k=8, k_heads=3, seed=0,
    )
    # ablation: 2 seqs x 4 conditions = 8 rows
    assert len(ablation_df) == 8
    assert set(ablation_df["condition"].unique()) == {
        "unmasked", "cycle", "random", "magnitude"}
    assert ablation_df["second_copy_loss"].notna().all()
    # cycles: schema present (may be empty if no cycles in tiny graphs)
    for col in ("model", "seq", "layer", "head", "is_induction", "gap"):
        assert col in cycles_df.columns
    # parquets written
    assert cyc.exists() and abl.exists()
    pd.read_parquet(abl)
