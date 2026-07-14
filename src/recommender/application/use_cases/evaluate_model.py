from recommender.application.ports.metric_calculator import MetricCalculator
from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.metrics import Metrics


class EvaluateModel:
    """Caso de uso: avalia um modelo sobre o conjunto de teste.

    Depende da PORTA MetricCalculator — a matemática das métricas
    (precision@k, recall@k, ndcg) vive na implementação, não aqui.
    """

    def __init__(self, metric_calculator: MetricCalculator) -> None:
        self._metric_calculator = metric_calculator

    def execute(
        self,
        model: RecommendationModel,
        test_interactions: list[Interaction],
        k: int = 10,
    ) -> Metrics:
        return self._metric_calculator.calculate(model, test_interactions, k)
