import pytest

from recommender.domain.value_objects.score import Score


def test_creates_valid_score() -> None:
    assert Score(0.87).value == 0.87


def test_accepts_zero() -> None:
    assert Score(0.0).value == 0.0


def test_rejects_negative() -> None:
    with pytest.raises(ValueError):
        Score(-0.1)


def test_score_is_not_orderable_by_itself() -> None:
    # Ranquear é responsabilidade da Recommendation, não do Score.
    with pytest.raises(TypeError):
        _ = Score(0.9) > Score(0.3)
        
def test_scores_are_equal_by_value() -> None:
    assert Score(0.5) == Score(0.5)