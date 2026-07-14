from datetime import UTC, datetime

import pytest

from recommender.application.use_cases.evaluate_model import EvaluateModel
from recommender.application.use_cases.generate_recommendation import (
    GenerateRecommendation,
)
from recommender.application.use_cases.pre_process_dataset import PreProcessDataset
from recommender.application.use_cases.train_model import TrainModel
from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.metrics import Metrics
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


def _interaction(user: int = 1, item: int = 1, rating: float = 4.0, minute: int = 0) -> Interaction:
    return Interaction(
        user_id=UserId(user),
        item_id=ItemId(item),
        rating=Rating(rating),
        occurred_at=datetime(2024, 1, 1, 0, minute, tzinfo=UTC),
    )


# --- Fakes (test doubles que satisfazem as portas por estrutura) ---


class _FakeReader:
    def __init__(self, interactions: list[Interaction]) -> None:
        self._interactions = interactions

    def read_items(self) -> list:
        return []

    def read_interactions(self) -> list[Interaction]:
        return self._interactions


class _FakeModel:
    def __init__(self, items: list[RecommendedItem] | None = None) -> None:
        self._items = items or []

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        return Recommendation(user_id, self._items)


class _FakeTrainer:
    def __init__(self) -> None:
        self.called = False

    def train(self, interactions, hyperparameters) -> _FakeModel:
        self.called = True
        return _FakeModel()


class _FakeModelRepo:
    def __init__(self, model: _FakeModel | None = None) -> None:
        self._model = model
        self.saved: dict[str, object] = {}

    def save(self, name: str, model) -> None:
        self.saved[name] = model

    def get(self, name: str):
        return self._model


class _FakeInteractionRepo:
    def __init__(self, history: list[Interaction]) -> None:
        self._history = history

    def all(self) -> list[Interaction]:
        return self._history

    def for_user(self, user_id: UserId) -> list[Interaction]:
        return [i for i in self._history if i.user_id == user_id]


class _FakeCalculator:
    def calculate(self, model, test_interactions, k) -> Metrics:
        return Metrics({"precision_at_k": 0.5})


# --- Testes ---


def test_pre_process_splits_temporally() -> None:
    interactions = [_interaction(minute=m) for m in range(10)]
    reader = _FakeReader(list(reversed(interactions)))  # fora de ordem de propósito

    split = PreProcessDataset(reader).execute(test_ratio=0.2)

    assert len(split.train) == 8
    assert len(split.test) == 2
    assert split.train[0].occurred_at.minute == 0  # mais antigo no treino
    assert split.test[-1].occurred_at.minute == 9  # mais novo no teste


def test_pre_process_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        PreProcessDataset(_FakeReader([])).execute(test_ratio=1.5)


def test_train_model_trains_and_persists() -> None:
    trainer = _FakeTrainer()
    repo = _FakeModelRepo()

    model = TrainModel(trainer, repo).execute(
        model_name="m1",
        interactions=[_interaction()],
        hyperparameters=Hyperparameters({"lr": 0.1}),
    )

    assert trainer.called is True
    assert repo.saved["m1"] is model


def test_evaluate_model_delegates_to_calculator() -> None:
    metrics = EvaluateModel(_FakeCalculator()).execute(
        model=_FakeModel(), test_interactions=[_interaction()], k=5
    )

    assert metrics.get("precision_at_k") == 0.5


def test_generate_recommendation_filters_seen_and_returns_top_k() -> None:
    items = [
        RecommendedItem(ItemId(1), Score(0.9)),
        RecommendedItem(ItemId(2), Score(0.8)),
        RecommendedItem(ItemId(3), Score(0.7)),
        RecommendedItem(ItemId(4), Score(0.6)),
    ]
    model_repo = _FakeModelRepo(_FakeModel(items))
    history = [_interaction(user=7, item=2)]  # já viu o item 2

    recommendation = GenerateRecommendation(
        model_repo, _FakeInteractionRepo(history)
    ).execute(model_name="m", user_id=UserId(7), k=2, filter_seen=True)

    assert [ri.item_id.value for ri in recommendation.items] == [1, 3]


def test_generate_recommendation_raises_when_model_missing() -> None:
    with pytest.raises(ValueError):
        GenerateRecommendation(_FakeModelRepo(None), _FakeInteractionRepo([])).execute(
            model_name="missing", user_id=UserId(1)
        )
