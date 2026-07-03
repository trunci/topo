"""Unit tests for Exp 18 span features (no 7B model; tokenizer + numpy only)."""
import numpy as np
import pytest
from transformers import AutoTokenizer

from src.attn_extract import format_prompt
from src.span_features import paragraph_token_masks, span_head_features


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


PARAS = ("Alpha City: Alpha City is a town in Norland. It has 300 people.",
         "Beta Rock: Beta Rock is a famous rock band from Alpha City.",
         "Gamma Sea: The Gamma Sea borders no countries at all.")
IS_GOLD = (True, False, True)
QUESTION = "Where is the band Beta Rock from?"
PROMPT = ("Answer the following question based on the context.\n\n"
          "Context:\n" + "\n".join(PARAS) +
          f"\n\nQuestion: {QUESTION}\n\nAnswer in as few words as possible.")


def test_masks_cover_disjoint_regions(tok):
    formatted = format_prompt(tok, PROMPT)
    n = len(tok(formatted)["input_ids"])
    gold, dist, q = paragraph_token_masks(tok, PROMPT, PARAS, IS_GOLD,
                                          QUESTION, n)
    assert gold.any() and dist.any() and q.any()
    assert not (gold & dist).any()
    assert not (gold & q).any()
    assert not (dist & q).any()
    # decoding the masked tokens recovers the paragraph text
    ids = np.array(tok(formatted)["input_ids"])
    gold_text = tok.decode(ids[gold])
    assert "Alpha City is a town" in gold_text and "Gamma Sea" in gold_text
    assert "Beta Rock is a famous" not in gold_text
    dist_text = tok.decode(ids[dist])
    assert "Beta Rock is a famous" in dist_text


def test_span_head_features_mass_direction(tok):
    formatted = format_prompt(tok, PROMPT)
    n = len(tok(formatted)["input_ids"])
    gold, dist, q = paragraph_token_masks(tok, PROMPT, PARAS, IS_GOLD,
                                          QUESTION, n)
    R, heads = 3, 4
    rows = np.full((heads, R, n + R), 1e-6)
    gold_cols = np.where(gold)[0]
    dist_cols = np.where(dist)[0]
    rows[0, :, gold_cols[0]] = 0.9          # head 0 locks onto gold
    rows[1, :, dist_cols[0]] = 0.9          # head 1 locks onto distractor
    feats = span_head_features([rows], p=n, gold_mask=gold, dist_mask=dist,
                               q_mask=q)
    assert feats["ph_gold_mass"][0] > feats["ph_gold_mass"][1]
    assert feats["ph_dist_mass"][1] > feats["ph_dist_mass"][0]
    assert feats["ph_gold_max"][0] == pytest.approx(0.9)
    assert 0.0 <= feats["gold_frac_pooled"] <= 1.0
    assert len(feats["ph_mtd"]) == heads
