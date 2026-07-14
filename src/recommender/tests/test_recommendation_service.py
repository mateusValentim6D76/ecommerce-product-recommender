from datetime import UTC, datetime

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.services.recommendation_service import RecommendationService
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


def _seen(item: int) -> Interaction:
    return Interaction(
        user_id=UserId(1),
        item_id=ItemId(item),
        rating=Rating(4.0),
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_filter_already_seen_removes_history_items() -> None:
    recommendation = Recommendation(
        user_id=UserId(1),
        items=[
            RecommendedItem(ItemId(1), Score(0.9)),
            RecommendedItem(ItemId(2), Score(0.8)),
            RecommendedItem(ItemId(3), Score(0.7)),
        ],
    )
    history = [_seen(2)]

    result = RecommendationService().filter_already_seen(recommendation, history)

    assert [ri.item_id.value for ri in result.items] == [1, 3]


def test_filter_already_seen_keeps_all_when_no_history() -> None:
    recommendation = Recommendation(
        user_id=UserId(1),
        items=[RecommendedItem(ItemId(1), Score(0.9))],
    )

    result = RecommendationService().filter_already_seen(recommendation, [])

    assert len(result.items) == 1
