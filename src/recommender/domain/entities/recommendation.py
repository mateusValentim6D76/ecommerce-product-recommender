from dataclasses import dataclass

from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class RecommendedItem:
    """Um item recomendado com a relevância prevista pelo modelo"""

    item_id: ItemId
    score: Score

@dataclass(frozen=True)
class Recommendation:
    """Resultado do sistema: itens recomendados para um usuário.

    Não sabe de onde vieram os scores (baseline, MLP, embeddings)
    A política de ranqueamento (maior score primeiro) mora aqui"""

    user_id: UserId
    items: list[RecommendedItem]

    def ranked(self) -> list[RecommendedItem]:
        """Itens ordenados por score, do mais relevante ao menos"""
        return sorted(self.items, key=lambda recommended_item: recommended_item.score.value, reverse=True)

    def top_k(self, k: int) -> "Recommendation":
        """Nova Recommendation só com os k itens de maior score"""
        if k < 0:
            raise ValueError("k must be greater than or equal to zero")
        return Recommendation(user_id= self.user_id, items=self.ranked()[:k])