import pytest

from recommender.domain.value_objects.rating import Rating


def test_creates_valid_rating():
    assert Rating(3.5).value == 3.5


def test_accepts_boundaries() -> None:
    assert Rating(0.5).value == 0.5
    assert Rating(5.0).value == 5.0


def test_rejects_below_minimum() -> None:
    with pytest.raises(ValueError):
        Rating(0.0)


def test_rejects_above_maximum() -> None:
    with pytest.raises(ValueError):
        Rating(5.5)


def test_is_positive_with_default_threshold() -> None:
    assert Rating(4.0).is_positive() is True
    assert Rating(3.5).is_positive() is False


def test_is_positive_with_custom_threshold() -> None:
    assert Rating(3.0).is_positive(threshold=2.5) is True
    assert Rating(3.0).is_positive(threshold=4.0) is False
