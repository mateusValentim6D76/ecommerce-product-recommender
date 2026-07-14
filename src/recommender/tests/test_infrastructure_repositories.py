from datetime import UTC, datetime

from recommender.domain.entities.experiment import Experiment
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.metrics import Metrics
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.datasets.pandas_interaction_repository import (
    PandasInteractionRepository,
)
from recommender.infrastructure.persistence.file_pickle_model_repository import (
    FilePickleModelRepository,
)
from recommender.infrastructure.persistence.in_memory_experiment_repository import (
    InMemoryExperimentRepository,
)
from recommender.infrastructure.persistence.in_memory_model_repository import (
    InMemoryModelRepository,
)


class _PicklableModel:
    """Modelo mínimo (picklável) para testar persistência em disco."""

    def recommend(self, user_id: UserId, k: int):
        return None


def _interaction(user: int, item: int) -> Interaction:
    return Interaction(
        user_id=UserId(user),
        item_id=ItemId(item),
        rating=Rating(4.0),
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_in_memory_model_repository_save_and_get() -> None:
    repo = InMemoryModelRepository()
    model = _PicklableModel()

    assert repo.get("m") is None
    repo.save("m", model)
    assert repo.get("m") is model


def test_in_memory_experiment_repository_upsert() -> None:
    repo = InMemoryExperimentRepository()
    exp = Experiment("e1", "baseline", Hyperparameters({"k": 10}))

    repo.save(exp)
    assert repo.get_by_name("e1") is exp
    assert repo.get_by_name("nope") is None

    exp.finish(Metrics({"precision_at_k": 0.4}), datetime(2024, 1, 1, tzinfo=UTC))
    repo.save(exp)  # upsert pela identidade (name)
    assert len(repo.all()) == 1


def test_file_pickle_model_repository_roundtrip(tmp_path) -> None:
    repo = FilePickleModelRepository(tmp_path)

    assert repo.get("absent") is None
    repo.save("m", _PicklableModel())
    loaded = repo.get("m")

    assert isinstance(loaded, _PicklableModel)


def test_pandas_interaction_repository_queries() -> None:
    interactions = [_interaction(1, 10), _interaction(1, 20), _interaction(2, 30)]
    repo = PandasInteractionRepository(interactions)

    assert len(repo.all()) == 3
    assert len(repo.for_user(UserId(1))) == 2
    assert repo.for_user(UserId(99)) == []
