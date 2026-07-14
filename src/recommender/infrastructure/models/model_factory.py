from recommender.application.ports.model_trainer import ModelTrainer

_VALID_TYPES = ("baseline", "pytorch")


def create_trainer(model_type: str) -> ModelTrainer:
    """Factory Method: cria o ModelTrainer certo a partir de uma string.

    Concentra o if/else de seleção de modelo num lugar só — o resto do
    código pede 'baseline' ou 'pytorch' sem conhecer as classes concretas.

    Import é PREGUIÇOSO (dentro do if) de propósito: assim importar este
    módulo não puxa torch/sklearn; só a criação efetiva paga esse custo.
    """
    key = model_type.strip().lower()

    if key == "baseline":
        from recommender.infrastructure.models.sklearn_baseline_model import (
            BaselineTrainer,
        )

        return BaselineTrainer()

    if key == "pytorch":
        from recommender.infrastructure.training.pytorch_trainer import PyTorchTrainer

        return PyTorchTrainer()

    raise ValueError(
        f"Unknown model_type: {model_type!r}. Valid: {_VALID_TYPES}"
    )
