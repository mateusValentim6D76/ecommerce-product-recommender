from typing import Protocol, runtime_checkable

from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.user_id import UserId


@runtime_checkable
class InteractionRepository(Protocol):
    """Repositório: acesso a coleção de interações do domínio.

    Abstrai a origem (CSV, banco, memória). O domínio conversa
    com esta interface como se fosse uma coleção de Interaction.
    """

    def all(self) -> list[Interaction]:
        """Todas as interações disponíveis."""
        ...

    def for_user(self, user_id: UserId) -> list[Interaction]:
        """Interações de um usuário específico (lista vazia se não houver)."""
        ...
