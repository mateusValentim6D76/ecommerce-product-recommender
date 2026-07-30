from dataclasses import dataclass, field

from recommender.domain.value_objects.item_id import ItemId


@dataclass(frozen=True)
class Item:
    """Item do catálogo que pode ser recomendado.

    Entity: sua identidade é o item_id. No MovieLens é um filme,
    mas o domínio o trata como Item genérico (produto, curso, etc.)"""

    item_id: ItemId
    title: str = field(compare=False)
    genres: list[str] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Item title must not be empty")

    def has_genre(self, genre: str) -> bool:
        return genre in self.genres
