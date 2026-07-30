from datetime import UTC, datetime

import pytest

pytest.importorskip("torch")  # pula se torch ainda não foi instalado

from recommender.domain.value_objects.hyperparameters import (  # noqa: E402
    Hyperparameters,
)
from recommender.domain.value_objects.interaction import Interaction  # noqa: E402
from recommender.domain.value_objects.item_id import ItemId  # noqa: E402
from recommender.domain.value_objects.rating import Rating  # noqa: E402
from recommender.domain.value_objects.user_id import UserId  # noqa: E402
from recommender.infrastructure.training.pytorch_trainer import (  # noqa: E402
    PyTorchTrainer,
)

_HP = Hyperparameters({"epochs": 2, "embedding_dim": 4, "batch_size": 4})


def _dataset() -> list[Interaction]:
    rows = [(1, 1, 5.0), (1, 2, 4.0), (2, 1, 3.0), (2, 3, 5.0), (1, 3, 4.0)]
    return [
        Interaction(UserId(u), ItemId(i), Rating(r), datetime(2024, 1, 1, tzinfo=UTC))
        for u, i, r in rows
    ]


def test_trains_and_recommends_known_items() -> None:
    model = PyTorchTrainer().train(_dataset(), _HP)

    recommendation = model.recommend(UserId(1), k=2)

    assert len(recommendation.items) <= 2
    known = {1, 2, 3}
    assert all(ri.item_id.value in known for ri in recommendation.items)


def test_unknown_user_returns_empty() -> None:
    model = PyTorchTrainer().train(_dataset(), _HP)

    assert model.recommend(UserId(999), k=3).items == []
