from typing import Protocol, runtime_checkable

from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.metrics import Metrics


@runtime_checkable
class MetricCalculator(Protocol):
    """Porta: calcula métricas de avaliação de um modelo.

    A MATEMÁTICA das métricas (precision@k, recall@k, ndcg) é
    responsabilidade da implementação — este é apenas o contrato.
    """

    def calculate(
        self,
        model: RecommendationModel,
        test_interactions: list[Interaction],
        k: int,
    ) -> Metrics:
        """Avalia o modelo sobre as interações de teste e devolve Metrics."""
        ...
