from dataclasses import dataclass
from datetime import datetime

from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class Interaction:
    """Registro de que um usuário avaliou um item num instante
    Fato imutável (value object semanticamente): não tem id próprio,
    é definido pelos seus dados. É a unidade de dado de treino"""

    user_id: UserId
    item_id: ItemId
    rating: Rating
    occurred_at: datetime

    def is_positive(self, threshold: float = 4.0) -> bool:
        """Delegado ao Rating: a regra de 'positivo' mora lá."""
        return self.rating.is_positive(threshold)
