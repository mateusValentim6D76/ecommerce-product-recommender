from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class IndexEncoder(Generic[T]):
    """Mapeia ids esparsos (UserId/ItemId) para índices contíguos 0..N-1.

    As camadas de embedding do PyTorch precisam de índices densos
    (0, 1, 2, ...), mas os ids do MovieLens têm buracos. Este encoder
    faz a ponte nos dois sentidos: id <-> índice.

    Você vai USAR isto no modelo PyTorch (Passo de treino):
    - index_of(user_id) para montar os tensores de entrada;
    - id_of(index) para traduzir a saída do modelo de volta ao domínio.
    """

    def __init__(self) -> None:
        self._to_index: dict[T, int] = {}
        self._to_id: list[T] = []

    def fit(self, ids: Iterable[T]) -> "IndexEncoder[T]":
        for identifier in ids:
            if identifier not in self._to_index:
                self._to_index[identifier] = len(self._to_id)
                self._to_id.append(identifier)
        return self

    def index_of(self, identifier: T) -> int:
        return self._to_index[identifier]

    def id_of(self, index: int) -> T:
        return self._to_id[index]

    def __contains__(self, identifier: T) -> bool:
        return identifier in self._to_index

    def __len__(self) -> int:
        return len(self._to_id)
