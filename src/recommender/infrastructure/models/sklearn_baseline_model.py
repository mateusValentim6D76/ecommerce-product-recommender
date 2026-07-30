"""Baseline de popularidade: recomenda os itens mais avaliados positivamente."""

from collections import Counter

from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


class PopularityModel:
    """Recomenda a mesma lista de itens populares para qualquer usuário."""

    def __init__(self, ranked_items: list[RecommendedItem]) -> None:
        self._ranked_items = ranked_items

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        return Recommendation(user_id=user_id, items=self._ranked_items[:k])


class BaselineTrainer:
    def train(
        self,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> PopularityModel:
        counts = Counter(
            interaction.item_id
            for interaction in interactions
            if interaction.is_positive()
        )
        if not counts:
            return PopularityModel([])

        max_count = max(counts.values())
        ranked_items = [
            RecommendedItem(item_id=item_id, score=Score(count / max_count))
            for item_id, count in counts.most_common()
        ]
        return PopularityModel(ranked_items)
