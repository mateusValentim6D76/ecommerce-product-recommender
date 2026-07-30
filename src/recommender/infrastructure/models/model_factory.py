"""Factory de trainers de modelo (imports preguiçosos por tipo)."""

from recommender.application.ports.model_trainer import ModelTrainer

_VALID_TYPES = ("baseline", "pytorch")


def create_trainer(model_type: str) -> ModelTrainer:
    key = model_type.strip().lower()

    if key == "baseline":
        from recommender.infrastructure.models.sklearn_baseline_model import (
            BaselineTrainer,
        )

        return BaselineTrainer()

    if key == "pytorch":
        from recommender.infrastructure.training.pytorch_trainer import PyTorchTrainer

        return PyTorchTrainer()

    raise ValueError(f"Unknown model_type: {model_type!r}. Valid: {_VALID_TYPES}")
