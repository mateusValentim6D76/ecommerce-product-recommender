from dataclasses import dataclass

from recommender.domain.value_objects.interaction import Interaction


@dataclass(frozen=True)
class DatasetSplit:
    """Resultado do pré-processamento: partições de treino e teste."""

    train: list[Interaction]
    test: list[Interaction]
