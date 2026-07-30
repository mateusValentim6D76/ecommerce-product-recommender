"""Encoder de ids esparsos para índices densos (exigência das embeddings)."""

from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class IndexEncoder(Generic[T]):
    """Mapeia ids para índices contíguos 0..N-1 e vice-versa."""

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
