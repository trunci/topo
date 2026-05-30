import pytest


def test_methods_constant_lists_all_five():
    from src.run_exp1 import METHODS
    assert METHODS == ["unpruned", "dmt", "magnitude", "random", "window"]


def test_keepsets_for_methods_match_dmt_budget():
    # On a synthetic attention matrix, baselines must keep exactly as many edges
    # as DMT does (budget matching), and unpruned keeps all candidate edges.
    import numpy as np
    from src.run_exp1 import keepsets_for_example
    rng = np.random.default_rng(0)
    # one layer, two heads, n=6
    att = rng.random((1, 2, 6, 6))
    per_method = keepsets_for_example(att, top_k=4, seed=0)
    for (li, h) in [(0, 0), (0, 1)]:
        kd = len(per_method["dmt"][(li, h)])
        assert len(per_method["magnitude"][(li, h)]) == kd
        assert len(per_method["random"][(li, h)]) == kd
        assert len(per_method["window"][(li, h)]) == kd
        assert kd > 0
