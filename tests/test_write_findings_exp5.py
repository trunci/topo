import json

from src.write_findings_exp5 import render, main


_STATS = {
    "model": "gpt2", "seq_len": 25, "gap_tolerance": 2, "alpha": 0.05,
    "E2": {
        "metric": "fraction of cycle edges with |gap - S| <= 2 (S=25)",
        "n_induction_heads": 2, "n_noninduction_heads": 2,
        "induction_fractions": [1.0, 0.5],
        "noninduction_fractions": [0.0, 0.0],
        "induction_mean_fraction": 0.75, "noninduction_mean_fraction": 0.0,
        "gap_histogram_induction": {"24": 1, "25": 2},
        "gap_histogram_noninduction": {"1": 3},
        "mann_whitney_p": 0.04, "verdict": "GREEN",
    },
    "E1": {
        "n_sequences": 8,
        "median_damage": {"cycle": 1.0, "random": 0.2, "magnitude": 0.5},
        "cycle_vs_magnitude": {"p_value": 0.01, "median_diff": 0.5},
        "cycle_vs_random": {"p_value": 0.005, "median_diff": 0.8},
        "sanity": {"cycle_damage_positive": True,
                   "magnitude_damage_positive": True, "note": "x"},
        "verdict": "GREEN",
    },
}


def test_render_contains_verdicts_and_numbers_from_json():
    md = render(_STATS)
    assert "Verdict: GREEN" in md
    assert "0.75" in md          # induction mean fraction
    assert "1.0000" in md        # cycle median damage
    assert "gpt2" in md


def test_main_writes_byte_identical_on_regeneration(tmp_path):
    sp = tmp_path / "stats.json"
    out = tmp_path / "FINDINGS.md"
    json.dump(_STATS, open(sp, "w"), indent=2)
    a = main(str(sp), str(out))
    first = open(out).read()
    b = main(str(sp), str(out))
    second = open(out).read()
    assert a == b == first == second
