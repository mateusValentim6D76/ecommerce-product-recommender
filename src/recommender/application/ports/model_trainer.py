from typing import Protocol, runtime_checkable

from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction


@runtime_checkable
class ModelTrainer(Protocol):
    """Porta: contrato para treinar um modelo.

    Recebe interações + hiperparâmetros e devolve um
    RecommendationModel já treinado. A implementação concreta
    (baseline, PyTorch) é responsabilidade da infraestrutura.
    """

    def train(
        self,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> RecommendationModel:
        """Treina e retorna um modelo pronto para recomendar."""
        ...
