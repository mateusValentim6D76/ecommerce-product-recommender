import pytest

from recommender.domain.value_objects.item_id import ItemId

def test_creates_valid_item_id() -> None:
    item_id = ItemId(10)

    assert item_id.value == 10


def test_two_item_ids_with_same_value_are_equal() -> None:
    assert ItemId(10) == ItemId(10)

def test_is_immutable() -> None:
    item_id = ItemId(10)

    with pytest.raises(Exception):
        item_id.value = 20
        

def test_rejects_zero() -> None:
    with pytest.raises(ValueError):
        ItemId(0)


def test_rejects_negative() -> None:
    with pytest.raises(ValueError):
        ItemId(-5)