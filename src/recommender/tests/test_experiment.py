from datetime import datetime

from recommender.domain.entities.experiment import Experiment
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.metrics import Metrics


def _experiment() -> Experiment:
    return Experiment(
        name="baseline-v1",
        model_name="popularity",
        hyperparameters=Hyperparameters({"top_k": 10}),
    )


def test_new_experiment_is_not_finished() -> None:
    exp = _experiment()

    assert exp.is_finished is False
    assert exp.metrics is None


def test_finish_sets_metrics_and_timestamp() -> None:
    exp = _experiment()
    metrics = Metrics({"precision_at_10": 0.42})
    finished_at = datetime(2024, 1, 1, 15, 0, 0)

    exp.finish(metrics=metrics, finished_at=finished_at)

    assert exp.is_finished is True
    assert exp.metrics.get("precision_at_10") == 0.42
    assert exp.finished_at == finished_at


def test_hyperparameters_are_accessible() -> None:
    exp = _experiment()

    assert exp.hyperparameters.get("top_k") == 10


def test_identity_is_by_name_regardless_of_state() -> None:
    a = Experiment("baseline-v1", "popularity", Hyperparameters({"top_k": 10}))
    b = Experiment("baseline-v1", "mlp", Hyperparameters({"lr": 0.01}))

    # Mesmo nome, model_name/hp diferentes -> mesmo experimento
    assert a == b


def test_finishing_does_not_change_identity() -> None:
    a = Experiment("baseline-v1", "popularity", Hyperparameters({"top_k": 10}))
    b = Experiment("baseline-v1", "popularity", Hyperparameters({"top_k": 10}))

    a.finish(Metrics({"precision_at_10": 0.42}), datetime(2024, 1, 1))

    # a terminou, b não masss continuam sendo o mesmo experimento
    assert a == b
    assert hash(a) == hash(b)