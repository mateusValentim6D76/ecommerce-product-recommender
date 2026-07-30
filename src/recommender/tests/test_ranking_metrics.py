from datetime import UTC, datetime

import pytest

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.evaluation.ranking_metrics import RankingMetricCalculator

A, C, X = ItemId(1), ItemId(3), ItemId(9)


def _interaction(item: ItemId, rating: float) -> Interaction:
    return Interaction(
        user_id=UserId(1),
        item_id=item,
        rating=Rating(rating),
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


class _FixedModel:
    """Modelo-fake: recomenda sempre [A, X, C] (X não é relevante)."""

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        return Recommendation(
            user_id=user_id,
            items=[
                RecommendedItem(A, Score(0.9)),
                RecommendedItem(X, Score(0.5)),
                RecommendedItem(C, Score(0.3)),
            ],
        )


def test_aggregates_metrics_over_users() -> None:
    # relevante do user 1 = {A, C} (item B com nota 2.0 é ignorado)
    interactions = [
        _interaction(A, 5.0),
        _interaction(C, 4.0),
        _interaction(ItemId(2), 2.0),  # negativo -> não conta
    ]

    metrics = RankingMetricCalculator().calculate(_FixedModel(), interactions, k=3)

    assert metrics.get("precision_at_k") == pytest.approx(2 / 3)
    assert metrics.get("recall_at_k") == pytest.approx(1.0)
    assert metrics.get("ndcg_at_k") == pytest.approx(0.9197207)


def test_returns_zeros_when_no_positive_interactions() -> None:
    metrics = RankingMetricCalculator().calculate(_FixedModel(), [_interaction(A, 1.0)], k=3)

    assert metrics.get("precision_at_k") == 0.0
    assert metrics.get("recall_at_k") == 0.0
    assert metrics.get("ndcg_at_k") == 0.0
