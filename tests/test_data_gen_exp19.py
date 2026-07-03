"""Unit tests for the Exp 19 rung generators and rewrite validators."""
import random

from src.data_gen_exp19 import (KIN_REL, _names_of, build_r1, build_r2,
                                rewrite_context_prompt, validate_context,
                                validate_question)


def test_r2_preserves_semantics_changes_surface():
    r1 = build_r1(n_per_family=5)
    r2 = build_r2(n_per_family=5)
    assert len(r1) == len(r2) == 20
    changed = 0
    for a, b in zip(r1, r2):
        assert b.id == a.id + "-r2"
        assert b.gold == a.gold and b.hop == a.hop and b.family == a.family
        assert _names_of(a) == _names_of(b.__class__(**{**b.__dict__,
                                                        "context": a.context}))
        # every cast name survives in the paraphrased context
        for n in _names_of(a):
            assert n in b.context
        changed += int(b.context != a.context or b.question != a.question)
    assert changed >= len(r1) * 0.9  # surface actually varies


def test_names_of_recovers_chain_order():
    for it in build_r1(n_per_family=3):
        names = _names_of(it)
        assert len(names) == it.hop + 1
        assert it.gold in names


def test_context_validator():
    it = build_r1(n_per_family=1)[0]
    names = _names_of(it)
    good = ("The family story begins with " + names[0] + ", whose line runs "
            + " then ".join(names[1:]) + ", father to child each time.")
    assert validate_context(it, good)
    assert not validate_context(it, good.replace(names[2], "Bob"))  # lost name
    rel = KIN_REL[it.hop]
    assert not validate_context(it, good + f" So the {rel} is obvious.")


def test_question_validator():
    it = [i for i in build_r1(n_per_family=1) if i.family == "kinship"][0]
    subj = _names_of(it)[-1]
    rel = KIN_REL[it.hop]
    assert validate_question(it, f"So tell me, who exactly is {subj}'s {rel}?")
    assert not validate_question(it, f"Who is {subj}'s ancestor?")  # rel gone
    assert not validate_question(it, f"Who is Bob's {rel}?")        # subj gone
    assert not validate_question(it, f"Tell me {subj}'s {rel}.")    # no '?'


def test_rewrite_prompt_contains_facts():
    it = build_r1(n_per_family=1)[0]
    assert it.context in rewrite_context_prompt(it)
