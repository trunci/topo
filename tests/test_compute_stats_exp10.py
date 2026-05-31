import numpy as np
import pandas as pd
from src.compute_stats_exp10 import _part_a, _part_b


def _attr(n, topo_good, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        mag = rng.uniform(0.7, 0.9)
        cyc = rng.uniform(0.85, 0.97) if topo_good else rng.uniform(0.45, 0.55)
        for sal, a in [("magnitude", mag), ("cycle_participation", cyc),
                       ("sheaf_discord", rng.uniform(0.45, 0.55))]:
            rows.append({"prompt": f"p{i}", "layer": 0, "head": 0, "saliency": sal,
                         "auc": a, "p_at_1": a})
    return pd.DataFrame(rows)


def test_part_a_green_when_topology_beats_magnitude():
    assert _part_a(_attr(40, True, 1))["verdict"] == "GREEN"


def test_part_a_not_green_when_topology_chance():
    assert _part_a(_attr(40, False, 2))["verdict"] in ("PARTIAL", "RED")


def _abl(n, salient_hurts, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        un = 3.0
        cyc = un - rng.uniform(0.5, 1.0) if salient_hurts else un - rng.uniform(-0.05, 0.05)
        rows += [
            {"prompt": f"p{i}", "condition": "unmasked", "io_margin": un},
            {"prompt": f"p{i}", "condition": "cycle", "io_margin": cyc},
            {"prompt": f"p{i}", "condition": "sheaf", "io_margin": cyc},
            {"prompt": f"p{i}", "condition": "magnitude", "io_margin": un - rng.uniform(0.5, 1.0)},
            {"prompt": f"p{i}", "condition": "random", "io_margin": un - rng.uniform(-0.05, 0.05)},
        ]
    return pd.DataFrame(rows)


def test_part_b_green_when_salient_hurts():
    b = _part_b(_abl(20, True, 3))
    assert b["cycle_damage_positive"] is True and b["verdict"] == "GREEN"


def test_part_b_not_green_when_no_damage():
    assert _part_b(_abl(20, False, 4))["verdict"] in ("RED", "PARTIAL")
