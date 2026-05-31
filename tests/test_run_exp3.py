import pytest
import pandas as pd


@pytest.mark.slow
def test_run_exp3_writes_per_head_parquet(tmp_path):
    from src.run_exp3 import run
    out = tmp_path / "exp3.parquet"
    run(out_path=str(out), seq_len=8, prefix_len=2, n_seqs=2, top_k=8)
    df = pd.read_parquet(out)
    assert len(df) == 144
    expected = {"layer", "head", "induction_score", "h1_persistence",
                "attn_distance", "offdiag_mass", "attn_entropy"}
    assert expected.issubset(set(df.columns))
    assert df["induction_score"].notna().all()
    assert df["attn_entropy"].notna().all()
