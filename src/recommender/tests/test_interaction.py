from datetime import datetime

from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId


def _interaction(rating: float) -> Interaction:
    return Interaction(
        user_id=UserId(1),
        item_id=ItemId(10),
        rating=Rating(rating),
        occurred_at=datetime(2024, 1, 1, 12, 0, 0),
    )


def test_creates_interaction() -> None:
    interaction = _interaction(4.0)

    assert interaction.user_id == UserId(1)
    assert interaction.item_id == ItemId(10)
    assert interaction.rating == Rating(4.0)


def test_is_positive_delegates_to_rating() -> None:
    assert _interaction(4.0).is_positive() is True
    assert _interaction(3.0).is_positive() is False


def test_is_positive_respects_custom_threshold() -> None:
    assert _interaction(3.0).is_positive(threshold=2.5) is True


def test_two_identical_interactions_are_equal() -> None:
    assert _interaction(4.0) == _interaction(4.0)
