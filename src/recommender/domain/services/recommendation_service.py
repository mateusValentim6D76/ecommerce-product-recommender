from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId


class RecommendationService:
    """Domain Service: regras de recomendação que não pertencem a
    uma única entity/value object.

    Existe porque 'filtrar itens que o usuário já viu' cruza dados
    (a recomendação + o histórico de interações) — não cabe dentro
    de Recommendation nem de Interaction sozinhas.
    """

    def filter_already_seen(
        self,
        recommendation: Recommendation,
        history: list[Interaction],
    ) -> Recommendation:
        """Remove da recomendação os itens que o usuário já interagiu."""
        seen: set[ItemId] = {interaction.item_id for interaction in history}
        remaining: list[RecommendedItem] = [
            item for item in recommendation.items if item.item_id not in seen
        ]
        return Recommendation(user_id=recommendation.user_id, items=remaining)
