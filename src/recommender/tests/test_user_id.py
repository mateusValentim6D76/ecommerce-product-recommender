import pytest

from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.user_id import UserId


def test_creates_valid_user_id() -> None:
    user_id = UserId(7)

    assert user_id.value == 7


def test_two_user_ids_with_same_value_are_equal() -> None:
    assert UserId(7) == UserId(7)


def test_is_immutable() -> None:
    user_id = UserId(7)

    with pytest.raises(Exception):  # frozen dataclass -> FrozenInstanceError
        user_id.value = 20


def test_rejects_zero() -> None:
    with pytest.raises(ValueError):
        UserId(0)


def test_rejects_negative() -> None:
    with pytest.raises(ValueError):
        UserId(-1)


def test_user_id_and_item_id_are_not_equal_even_with_same_value() -> None:
    assert UserId(1) != ItemId(1)
