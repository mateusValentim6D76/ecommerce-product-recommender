from dataclasses import dataclass

from recommender.domain.entities.recommendation import Recommendation


@dataclass(frozen=True)
class RecommendedItemOutput:
    item_id: int
    score: float


@dataclass(frozen=True)
class RecommendationOutput:
    """Saída serializável de uma recomendação (para API/CLI).

    Converte os value objects do domínio (ItemId, Score) em
    primitivos, prontos para virar JSON. A fronteira da aplicação
    fala 'primitivo'; o domínio fala 'value object'.
    """

    user_id: int
    items: list[RecommendedItemOutput]

    @classmethod
    def from_domain(cls, recommendation: Recommendation) -> "RecommendationOutput":
        return cls(
            user_id=recommendation.user_id.value,
            items=[
                RecommendedItemOutput(
                    item_id=item.item_id.value,
                    score=item.score.value,
                )
                for item in recommendation.items
            ],
        )
