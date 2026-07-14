"""[VOCÊ IMPLEMENTA] Calculadora de métricas de ranking.

Satisfaz a porta MetricCalculator. Orquestra as funções puras de
`metric_strategy` sobre o conjunto de teste e devolve um Metrics agregado
(médias sobre os usuários).
"""

from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.metrics import Metrics


class RankingMetricCalculator:
    def calculate(
        self,
        model: RecommendationModel,
        test_interactions: list[Interaction],
        k: int,
    ) -> Metrics:
        # PASSO A PASSO sugerido:
        # 1. Monte, por usuário, o conjunto de itens RELEVANTES no teste
        #    (ex.: interações positivas -> relevant[user_id].add(item_id)).
        # 2. Para cada usuário: rec = model.recommend(user_id, k)
        #    recommended_ids = [ri.item_id for ri in rec.ranked()]
        # 3. Calcule precision@k, recall@k, ndcg@k (funções de metric_strategy).
        # 4. Faça a MÉDIA de cada métrica sobre todos os usuários avaliados.
        # 5. Devolva Metrics({"precision_at_k": ..., "recall_at_k": ...,
        #                     "ndcg_at_k": ...}).
        raise NotImplementedError("Implemente o cálculo agregado das métricas")
