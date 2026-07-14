import pytest

from recommender.domain.value_objects.item_id import ItemId
from recommender.infrastructure.evaluation.metric_strategy import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Itens de apoio: A=1, B=2, C=3, D=4, E=5
A, B, C, D, E = (ItemId(i) for i in range(1, 6))


# Cenário calculado à mão:
#   recomendado = [A, B, C, D], relevante = {A, C, E}, k = 4
#   acertos nos top-4 = {A, C} -> 2
#   precision = 2/4 = 0.5
#   recall    = 2/3 ≈ 0.6667
#   DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5   (A na pos 0, C na pos 2)
#   IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) ≈ 2.1309 (3 acertos ideais)
#   ndcg = 1.5 / 2.1309 ≈ 0.7039


def test_precision_is_hits_over_k() -> None:
    assert precision_at_k([A, B, C, D], {A, C, E}, 4) == 0.5


def test_recall_is_hits_over_total_relevant() -> None:
    assert recall_at_k([A, B, C, D], {A, C, E}, 4) == pytest.approx(2 / 3)


def test_ndcg_rewards_hits_near_the_top() -> None:
    assert ndcg_at_k([A, B, C, D], {A, C, E}, 4) == pytest.approx(0.70391807)


def test_precision_with_non_positive_k_is_zero() -> None:
    assert precision_at_k([A], {A}, 0) == 0.0


def test_metrics_with_empty_relevant_are_zero() -> None:
    assert recall_at_k([A, B], set(), 2) == 0.0
    assert ndcg_at_k([A, B], set(), 2) == 0.0


def test_perfect_ranking_scores_one() -> None:
    assert precision_at_k([A, B, C], {A, B}, 2) == 1.0
    assert recall_at_k([A, B, C], {A, B}, 2) == 1.0
    assert ndcg_at_k([A, B, C], {A, B}, 2) == pytest.approx(1.0)


def test_slicing_handles_lists_shorter_than_k() -> None:
    # k=5 mas só há 1 recomendado: [:5] devolve os que existem, sem erro.
    assert precision_at_k([A], {A}, 5) == pytest.approx(0.2)
    assert recall_at_k([A], {A}, 5) == 1.0
