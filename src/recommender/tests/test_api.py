import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from recommender.domain.entities.recommendation import (  # noqa: E402
    Recommendation,
    RecommendedItem,
)
from recommender.domain.value_objects.item_id import ItemId  # noqa: E402
from recommender.domain.value_objects.score import Score  # noqa: E402
from recommender.domain.value_objects.user_id import UserId  # noqa: E402
from recommender.interfaces.api.app import create_app  # noqa: E402


class _FakeGenerate:
    def execute(self, model_name: str, user_id: UserId, k: int) -> Recommendation:
        return Recommendation(user_id, [RecommendedItem(ItemId(1), Score(0.9))])


def _client() -> TestClient:
    return TestClient(create_app(_FakeGenerate()))


def test_health_endpoint() -> None:
    assert _client().get("/health").json() == {"status": "ok"}


def test_recommendations_endpoint_returns_items() -> None:
    response = _client().get("/users/7/recommendations?k=1")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 7
    assert body["items"][0]["item_id"] == 1
