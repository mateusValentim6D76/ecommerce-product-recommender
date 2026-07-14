from recommender.infrastructure.training.early_stopping import EarlyStopping


def test_stops_after_patience_without_improvement() -> None:
    stopper = EarlyStopping(patience=2)

    assert stopper.should_stop(1.0) is False  # primeira loss
    assert stopper.should_stop(0.9) is False  # melhora
    assert stopper.should_stop(0.95) is False  # sem melhora (1)
    assert stopper.should_stop(0.96) is True  # sem melhora (2) -> para


def test_resets_counter_on_improvement() -> None:
    stopper = EarlyStopping(patience=2)

    stopper.should_stop(1.0)
    stopper.should_stop(0.99)  # melhora -> reseta
    assert stopper.should_stop(1.0) is False  # sem melhora (1)
    assert stopper.should_stop(0.5) is False  # melhora -> reseta
    assert stopper.should_stop(0.6) is False  # sem melhora (1)
    assert stopper.should_stop(0.7) is True  # sem melhora (2) -> para


def test_tracks_best_loss() -> None:
    stopper = EarlyStopping(patience=5)

    stopper.should_stop(1.0)
    stopper.should_stop(0.3)
    stopper.should_stop(0.8)

    assert stopper.best_loss == 0.3
