import pytest

from recommender.domain.entities.item import Item
from recommender.domain.value_objects.item_id import ItemId


def test_creates_item() -> None:
    item = Item(ItemId(1), "Toy Story", ["Animation", "Comedy"])

    assert item.title == "Toy Story"
    assert item.has_genre("Comedy") is True
    assert item.has_genre("Horror") is False


def test_item_without_genres_defaults_to_empty_list() -> None:
    item = Item(ItemId(1), "Toy Story")

    assert item.genres == []


def test_two_items_with_same_data_are_equal() -> None:
    a = Item(ItemId(1), "Toy Story", ["Comedy"])
    b = Item(ItemId(1), "Toy Story", ["Comedy"])

    assert a == b


def test_rejects_empty_title() -> None:
    with pytest.raises(ValueError):
        Item(ItemId(1), "   ")


def test_default_genres_are_not_shared_between_items() -> None:
    a = Item(ItemId(1), "A")
    b = Item(ItemId(2), "B")

    assert a.genres is not b.genres


def test_identity_is_based_only_on_item_id() -> None:
    # Mesmo id, atributos diferentes -> mesmo item (entity)
    a = Item(ItemId(1), "Toy Story", ["Comedy"])
    b = Item(ItemId(1), "Toy Story (1995)", ["Animation"])

    assert a == b
