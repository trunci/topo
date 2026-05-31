import pytest
import pandas as pd


@pytest.mark.slow
def test_run_exp4_writes_per_head_parquet_both_models(tmp_path):
    from src.run_exp4 import run
    out = tmp_path / "exp4.parquet"
    run(out_path=str(out), seq_len=8, prefix_len=2, n_seqs=2, top_k=8)
    df = pd.read_parquet(out)
    expected = {"model", "layer", "head", "induction_score", "prev_token_score",
                "dup_token_score", "h1_persistence", "attn_distance",
                "offdiag_mass", "attn_entropy"}
    assert expected.issubset(set(df.columns))
    assert set(df["model"].unique()) == {"gpt2", "distilgpt2"}
    assert df["induction_score"].notna().all()
    assert df["prev_token_score"].notna().all()
    assert df["dup_token_score"].notna().all()
