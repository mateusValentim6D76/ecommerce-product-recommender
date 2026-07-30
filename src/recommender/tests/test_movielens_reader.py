import pytest

pytest.importorskip("pandas")  # pula se pandas ainda não foi instalado

from recommender.infrastructure.datasets.movielens_reader import (  # noqa: E402
    MovieLensReader,
)


def _write_movielens(tmp_path):
    movies = tmp_path / "movies.csv"
    movies.write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Comedy\n"
        "2,Jumanji (1995),(no genres listed)\n",
        encoding="utf-8",
    )
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "userId,movieId,rating,timestamp\n1,1,4.0,964982703\n1,2,3.5,964982931\n",
        encoding="utf-8",
    )
    return ratings, movies


def test_reads_items_with_parsed_genres(tmp_path) -> None:
    ratings, movies = _write_movielens(tmp_path)
    reader = MovieLensReader(ratings, movies)

    items = reader.read_items()

    assert items[0].title.startswith("Toy Story")
    assert items[0].genres == ["Adventure", "Comedy"]
    assert items[1].genres == []  # "(no genres listed)" -> vazio


def test_reads_interactions_as_domain_objects(tmp_path) -> None:
    ratings, movies = _write_movielens(tmp_path)
    reader = MovieLensReader(ratings, movies)

    interactions = reader.read_interactions()

    assert len(interactions) == 2
    assert interactions[0].rating.value == 4.0
    assert interactions[0].user_id.value == 1
    assert interactions[0].is_positive() is True
