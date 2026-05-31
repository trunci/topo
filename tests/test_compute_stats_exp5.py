import json

import numpy as np
import pandas as pd
import pytest

from src.compute_stats_exp5 import compute_stats


def _write_ablation(path, unmasked, cycle, random_, magnitude):
    rows = []
    for s, (u, c, r, m) in enumerate(zip(unmasked, cycle, random_, magnitude)):
        rows.append({"model": "gpt2", "seq": s, "condition": "unmasked",
                     "second_copy_loss": u})
        rows.append({"model": "gpt2", "seq": s, "condition": "cycle",
                     "second_copy_loss": c})
        rows.append({"model": "gpt2", "seq": s, "condition": "random",
                     "second_copy_loss": r})
        rows.append({"model": "gpt2", "seq": s, "condition": "magnitude",
                     "second_copy_loss": m})
    pd.DataFrame(rows).to_parquet(path)


def _write_cycles(path, ind_gaps, nonind_gaps):
    rows = []
    for h, gaps in enumerate(ind_gaps):
        for g in gaps:
            rows.append({"model": "gpt2", "seq": 0, "layer": 0, "head": h,
                         "is_induction": 1, "gap": g})
    for h, gaps in enumerate(nonind_gaps):
        for g in gaps:
            rows.append({"model": "gpt2", "seq": 0, "layer": 1, "head": h,
                         "is_induction": 0, "gap": g})
    pd.DataFrame(rows).to_parquet(path)


def test_e1_green_when_cycle_damage_exceeds_both(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    rng = np.random.default_rng(0)
    unmasked = np.full(n, 3.0)
    # cycle damage clearly > magnitude > random > 0
    cycle = unmasked + 1.0 + rng.normal(0, 0.01, n)
    magnitude = unmasked + 0.5 + rng.normal(0, 0.01, n)
    random_ = unmasked + 0.2 + rng.normal(0, 0.01, n)
    _write_ablation(abl, unmasked, cycle, random_, magnitude)
    _write_cycles(cyc, ind_gaps=[[25]], nonind_gaps=[[1]])
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    assert res["E1"]["verdict"] == "GREEN"
    assert res["E1"]["cycle_vs_magnitude"]["p_value"] < 0.05
    assert res["E1"]["cycle_vs_random"]["p_value"] < 0.05
    assert res["E1"]["sanity"]["cycle_damage_positive"] is True
    assert res["E1"]["sanity"]["magnitude_damage_positive"] is True


def test_e1_red_when_cycle_equals_random(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    unmasked = np.full(n, 3.0)
    # cycle == random == magnitude (same damage) -> no signal beyond random
    same = unmasked + 0.5
    _write_ablation(abl, unmasked, same.copy(), same.copy(), same.copy())
    _write_cycles(cyc, ind_gaps=[[25]], nonind_gaps=[[1]])
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    assert res["E1"]["verdict"] == "RED"


def test_e1_partial_when_cycle_beats_random_only(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    rng = np.random.default_rng(1)
    unmasked = np.full(n, 3.0)
    # cycle ~ magnitude (no reliable diff) but both > random
    cycle = unmasked + 0.8 + rng.normal(0, 0.01, n)
    magnitude = unmasked + 0.8 + rng.normal(0, 0.01, n)
    random_ = unmasked + 0.2 + rng.normal(0, 0.01, n)
    _write_ablation(abl, unmasked, cycle, random_, magnitude)
    _write_cycles(cyc, ind_gaps=[[25]], nonind_gaps=[[1]])
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    assert res["E1"]["verdict"] == "PARTIAL"
    assert res["E1"]["cycle_vs_random"]["p_value"] < 0.05


def test_e2_green_when_induction_concentrates_at_gap_S(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    unmasked = np.full(n, 3.0)
    _write_ablation(abl, unmasked, unmasked + 0.5, unmasked + 0.3, unmasked + 0.4)
    # induction heads: all cycle edges at gap ~ S=25; non-induction: gap ~1
    ind = [[25, 24, 26] for _ in range(6)]
    nonind = [[1, 2, 1] for _ in range(6)]
    _write_cycles(cyc, ind_gaps=ind, nonind_gaps=nonind)
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    assert res["E2"]["verdict"] == "GREEN"
    assert res["E2"]["mann_whitney_p"] < 0.05


def test_e2_red_when_no_concentration_difference(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    unmasked = np.full(n, 3.0)
    _write_ablation(abl, unmasked, unmasked + 0.5, unmasked + 0.3, unmasked + 0.4)
    # both groups: gaps far from S -> equal (zero) fractions
    ind = [[1, 2, 3] for _ in range(6)]
    nonind = [[1, 2, 3] for _ in range(6)]
    _write_cycles(cyc, ind_gaps=ind, nonind_gaps=nonind)
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    assert res["E2"]["verdict"] == "RED"


def test_on_disk_json_equals_returned_dict(tmp_path):
    abl = tmp_path / "abl.parquet"
    cyc = tmp_path / "cyc.parquet"
    n = 8
    unmasked = np.full(n, 3.0)
    _write_ablation(abl, unmasked, unmasked + 1.0, unmasked + 0.2, unmasked + 0.5)
    _write_cycles(cyc, ind_gaps=[[25]], nonind_gaps=[[1]])
    out = tmp_path / "stats.json"
    res = compute_stats(str(cyc), str(abl), str(out), seq_len=25)
    on_disk = json.load(open(out))
    assert on_disk == res
