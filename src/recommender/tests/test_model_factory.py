import pytest

from recommender.application.ports.model_trainer import ModelTrainer
from recommender.infrastructure.models.model_factory import create_trainer


def test_create_baseline_trainer_satisfies_port() -> None:
    trainer = create_trainer("baseline")

    # tipagem estrutural: tem 'train' -> é um ModelTrainer
    assert isinstance(trainer, ModelTrainer)


def test_create_trainer_is_case_insensitive() -> None:
    assert isinstance(create_trainer("BASELINE"), ModelTrainer)


def test_unknown_model_type_raises() -> None:
    with pytest.raises(ValueError):
        create_trainer("does-not-exist")
