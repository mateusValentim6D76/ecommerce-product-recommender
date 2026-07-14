from typing import Protocol, runtime_checkable

from recommender.domain.entities.recommendation import Recommendation
from recommender.domain.value_objects.user_id import UserId


@runtime_checkable
class RecommendationModel(Protocol):
    """Porta: um modelo TREINADO capaz de recomendar.

    É o artefato de inferência. O núcleo não sabe se por trás há
    um baseline de popularidade, uma MLP ou embeddings PyTorch —
    só sabe que, dado um usuário, ele devolve uma Recommendation.
    """

    def recommend(self, user_id: UserId, k: int) -> Recommendation:
        """Retorna os k itens mais relevantes para o usuário."""
        ...
