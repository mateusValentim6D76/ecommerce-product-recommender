from datetime import UTC, datetime

from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.preprocessing.movielens_preprocessor import IndexEncoder
from recommender.infrastructure.preprocessing.preprocessing_strategy import (
    ExplicitFeedbackStrategy,
    ImplicitFeedbackStrategy,
)


def _interaction(rating: float) -> Interaction:
    return Interaction(
        user_id=UserId(1),
        item_id=ItemId(1),
        rating=Rating(rating),
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_index_encoder_maps_ids_to_dense_indices() -> None:
    encoder = IndexEncoder().fit([UserId(10), UserId(20), UserId(10)])

    assert len(encoder) == 2
    assert encoder.index_of(UserId(10)) == 0
    assert encoder.index_of(UserId(20)) == 1
    assert encoder.id_of(0) == UserId(10)
    assert UserId(10) in encoder
    assert UserId(99) not in encoder


def test_explicit_feedback_uses_rating_value() -> None:
    assert ExplicitFeedbackStrategy().target(_interaction(3.5)) == 3.5


def test_implicit_feedback_thresholds() -> None:
    strategy = ImplicitFeedbackStrategy(threshold=4.0)

    assert strategy.target(_interaction(4.0)) == 1.0
    assert strategy.target(_interaction(3.0)) == 0.0
