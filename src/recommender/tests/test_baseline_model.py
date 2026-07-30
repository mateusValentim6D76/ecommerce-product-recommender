from datetime import UTC, datetime

import pytest

from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.models.sklearn_baseline_model import BaselineTrainer


def _interaction(item: int, rating: float) -> Interaction:
    return Interaction(
        user_id=UserId(1),
        item_id=ItemId(item),
        rating=Rating(rating),
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_baseline_ranks_items_by_popularity() -> None:
    interactions = [
        _interaction(1, 5.0),
        _interaction(1, 4.0),
        _interaction(1, 5.0),  # item 1: 3 positivos
        _interaction(2, 4.0),  # item 2: 1 positivo
        _interaction(3, 2.0),  # item 3: negativo -> não conta
    ]

    model = BaselineTrainer().train(interactions, Hyperparameters())
    recommendation = model.recommend(UserId(42), k=5)

    ids = [ri.item_id.value for ri in recommendation.items]
    assert ids == [1, 2]  # item 3 (negativo) fora
    assert recommendation.items[0].score.value == 1.0  # mais popular, score normalizado
    assert recommendation.items[1].score.value == pytest.approx(1 / 3)


def test_baseline_is_global_same_for_any_user() -> None:
    model = BaselineTrainer().train([_interaction(1, 5.0)], Hyperparameters())

    rec_a = model.recommend(UserId(1), k=1)
    rec_b = model.recommend(UserId(2), k=1)

    assert rec_a.items[0].item_id == rec_b.items[0].item_id


def test_baseline_with_no_positive_interactions_is_empty() -> None:
    model = BaselineTrainer().train([_interaction(1, 1.0)], Hyperparameters())

    assert model.recommend(UserId(1), k=5).items == []
