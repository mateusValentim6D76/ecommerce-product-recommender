"""Leitura dos CSVs do MovieLens, traduzindo cada linha em objetos do domínio."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from recommender.domain.entities.item import Item
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.rating import Rating
from recommender.domain.value_objects.user_id import UserId
from recommender.shared.exceptions import DatasetNotFoundError

_NO_GENRES = "(no genres listed)"


class MovieLensReader:
    """Implementa DatasetReader lendo movies.csv e ratings.csv com pandas."""

    def __init__(self, ratings_path: str | Path, movies_path: str | Path) -> None:
        self._ratings_path = Path(ratings_path)
        self._movies_path = Path(movies_path)

    def read_items(self) -> list[Item]:
        if not self._movies_path.exists():
            raise DatasetNotFoundError(f"movies file not found: {self._movies_path}")

        frame = pd.read_csv(self._movies_path)
        return [
            Item(
                item_id=ItemId(int(row.movieId)),
                title=str(row.title),
                genres=self._parse_genres(str(row.genres)),
            )
            for row in frame.itertuples(index=False)
        ]

    def read_interactions(self) -> list[Interaction]:
        if not self._ratings_path.exists():
            raise DatasetNotFoundError(f"ratings file not found: {self._ratings_path}")

        frame = pd.read_csv(self._ratings_path)
        return [
            Interaction(
                user_id=UserId(int(row.userId)),
                item_id=ItemId(int(row.movieId)),
                rating=Rating(float(row.rating)),
                occurred_at=datetime.fromtimestamp(int(row.timestamp), tz=timezone.utc),
            )
            for row in frame.itertuples(index=False)
        ]

    @staticmethod
    def _parse_genres(raw: str) -> list[str]:
        if raw == _NO_GENRES or not raw.strip():
            return []
        return raw.split("|")
