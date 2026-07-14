from typing import Protocol, runtime_checkable

from recommender.domain.entities.item import Item
from recommender.domain.value_objects.interaction import Interaction


@runtime_checkable
class DatasetReader(Protocol):
    """Porta: contrato para ler dados de um dataset de origem.

    O núcleo não sabe se por trás há CSV, Parquet ou banco.
    Qualquer objeto com estes métodos 'é' um DatasetReader.
    """

    def read_items(self) -> list[Item]:
        """Retorna o catálogo de itens."""
        ...

    def read_interactions(self) -> list[Interaction]:
        """Retorna as interações usuário-item."""
        ...
