"""Calculadora de métricas de ranking.

Implementa MetricCalculator: aplica as funções de `metric_strategy` por
usuário e devolve a média sobre os usuários avaliados.
"""

from collections import defaultdict

from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.metrics import Metrics
from recommender.domain.value_objects.user_id import UserId
from recommender.infrastructure.evaluation.metric_strategy import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class RankingMetricCalculator:
    def calculate(
        self,
        model: RecommendationModel,
        test_interactions: list[Interaction],
        k: int,
    ) -> Metrics:
        relevant_by_user: dict[UserId, set[ItemId]] = defaultdict(set)
        for interaction in test_interactions:
            if interaction.is_positive():
                relevant_by_user[interaction.user_id].add(interaction.item_id)

        if not relevant_by_user:
            return Metrics({"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0})

        total_precision = 0.0
        total_recall = 0.0
        total_ndcg = 0.0

        for user_id, relevant in relevant_by_user.items():
            recommendation = model.recommend(user_id, k)
            recommended_ids = [item.item_id for item in recommendation.ranked()]

            total_precision += precision_at_k(recommended_ids, relevant, k)
            total_recall += recall_at_k(recommended_ids, relevant, k)
            total_ndcg += ndcg_at_k(recommended_ids, relevant, k)

        evaluated_users = len(relevant_by_user)
        return Metrics(
            {
                "precision_at_k": total_precision / evaluated_users,
                "recall_at_k": total_recall / evaluated_users,
                "ndcg_at_k": total_ndcg / evaluated_users,
            }
        )
