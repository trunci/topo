import pandas as pd

from src.compute_stats_exp6 import _model_stats


def _df(losses_by_cond):
    """losses_by_cond: dict condition -> list of per-seq losses."""
    rows = []
    n = len(next(iter(losses_by_cond.values())))
    for cond, vals in losses_by_cond.items():
        for s, v in enumerate(vals):
            rows.append({"seq": s, "condition": cond, "second_copy_loss": v,
                         "k_ablated": 10})
    return pd.DataFrame(rows)


def test_green_when_cycle_positive_and_beats_random():
    # cycle clearly hurts (large positive damage), random barely
    df = _df({
        "unmasked": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cycle":    [3.0, 3.1, 2.9, 3.2, 3.0, 3.1],
        "magnitude":[3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "random":   [1.1, 1.0, 1.1, 1.0, 1.1, 1.0],
    })
    s = _model_stats(df)
    assert s["cycle_damage_positive"] is True
    assert s["beats_random"] is True
    assert s["verdict"] == "GREEN"
    assert s["median_damage"]["cycle"] > 1.5


def test_red_when_cycle_positive_but_not_above_random():
    # cycle hurts, but random hurts just as much -> not special
    df = _df({
        "unmasked": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cycle":    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "magnitude":[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "random":   [2.0, 2.1, 1.9, 2.0, 2.1, 1.9],
    })
    s = _model_stats(df)
    assert s["cycle_damage_positive"] is True
    assert s["beats_random"] is False
    assert s["verdict"] == "RED"


def test_inconclusive_when_cycle_damage_not_positive():
    # ablation barely moves / helps -> underpowered, like Exp 5 E1
    df = _df({
        "unmasked": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "cycle":    [0.98, 1.0, 0.97, 1.0, 0.99, 0.98],
        "magnitude":[0.98, 1.0, 0.97, 1.0, 0.99, 0.98],
        "random":   [0.99, 1.0, 0.99, 1.0, 1.0, 0.99],
    })
    s = _model_stats(df)
    assert s["cycle_damage_positive"] is False
    assert s["verdict"] == "INCONCLUSIVE"
