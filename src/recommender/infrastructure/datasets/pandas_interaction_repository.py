from collections.abc import Iterable

from recommender.application.ports.dataset_reader import DatasetReader
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.user_id import UserId


class PandasInteractionRepository:
    """Implementa InteractionRepository a partir de interações já lidas.

    Carrega as interações em memória (via um DatasetReader) e serve
    as consultas do domínio. Mantém a estrutura simples: uma lista
    de Interaction o nucleo nao sabe como isso é armazenado.
    """

    def __init__(self, interactions: Iterable[Interaction]) -> None:
        self._interactions: list[Interaction] = list(interactions)

    @classmethod
    def from_reader(cls, reader: DatasetReader) -> "PandasInteractionRepository":
        return cls(reader.read_interactions())

    def all(self) -> list[Interaction]:
        return list(self._interactions)

    def for_user(self, user_id: UserId) -> list[Interaction]:
        return [
            interaction
            for interaction in self._interactions
            if interaction.user_id == user_id
        ]
