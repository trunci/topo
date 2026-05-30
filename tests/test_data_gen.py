from src.data_gen import build_pairs, Item


def test_build_pairs_returns_items():
    items = build_pairs(n_per_family=5, seed=0)
    assert len(items) > 0
    assert all(isinstance(it, Item) for it in items)


def test_each_pair_has_one_1hop_and_one_2hop():
    items = build_pairs(n_per_family=5, seed=0)
    by_pair = {}
    for it in items:
        by_pair.setdefault(it.pair_id, []).append(it)
    for pid, group in by_pair.items():
        hops = sorted(it.hop for it in group)
        assert hops == [1, 2], f"pair {pid} has hops {hops}"


def test_pair_members_share_context_and_differ_in_question():
    items = build_pairs(n_per_family=5, seed=0)
    by_pair = {}
    for it in items:
        by_pair.setdefault(it.pair_id, []).append(it)
    for group in by_pair.values():
        one = next(it for it in group if it.hop == 1)
        two = next(it for it in group if it.hop == 2)
        # Shared leading context (everything up to the question) must match.
        assert one.context == two.context
        assert one.prompt != two.prompt


def test_gold_answers_appear_in_context():
    items = build_pairs(n_per_family=5, seed=0)
    for it in items:
        assert it.gold in it.context


def test_seed_is_deterministic():
    a = build_pairs(n_per_family=5, seed=42)
    b = build_pairs(n_per_family=5, seed=42)
    assert [it.prompt for it in a] == [it.prompt for it in b]
