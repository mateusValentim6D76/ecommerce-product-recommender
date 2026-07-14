from recommender.domain.entities.recommendation import (
    Recommendation,
    RecommendedItem,
)
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


def _rec(*pairs: tuple[int, float]) -> Recommendation:
    items = [RecommendedItem(ItemId(i), Score(s)) for i, s in pairs]
    return Recommendation(user_id=UserId(1), items=items)


def test_ranked_orders_by_score_descending() -> None:
    rec = _rec((1, 0.2), (2, 0.9), (3, 0.5))

    ranked_ids = [ri.item_id.value for ri in rec.ranked()]

    assert ranked_ids == [2, 3, 1]


def test_top_k_returns_highest_scores_in_order() -> None:
    rec = _rec((1, 0.2), (2, 0.9), (3, 0.5))

    top2 = rec.top_k(2)

    assert [ri.item_id.value for ri in top2.items] == [2, 3]


def test_top_k_does_not_mutate_original() -> None:
    rec = _rec((1, 0.2), (2, 0.9))

    rec.top_k(1)

    # original intacto (ordem de inserção preservada)
    assert [ri.item_id.value for ri in rec.items] == [1, 2]


def test_top_k_keeps_the_same_user() -> None:
    rec = _rec((1, 0.9))

    assert rec.top_k(1).user_id == UserId(1)