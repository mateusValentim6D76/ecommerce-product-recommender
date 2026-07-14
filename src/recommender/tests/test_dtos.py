from recommender.application.dtos.recommendation_output import RecommendationOutput
from recommender.application.dtos.train_model_input import TrainModelInput
from recommender.domain.entities.recommendation import Recommendation, RecommendedItem
from recommender.domain.value_objects.item_id import ItemId
from recommender.domain.value_objects.score import Score
from recommender.domain.value_objects.user_id import UserId


def test_recommendation_output_from_domain_converts_to_primitives() -> None:
    recommendation = Recommendation(
        user_id=UserId(5),
        items=[RecommendedItem(ItemId(10), Score(0.42))],
    )

    output = RecommendationOutput.from_domain(recommendation)

    assert output.user_id == 5
    assert output.items[0].item_id == 10
    assert output.items[0].score == 0.42


def test_train_model_input_defaults() -> None:
    payload = TrainModelInput(model_type="baseline", model_name="b1")

    assert payload.test_ratio == 0.2
    assert payload.k == 10
    assert payload.hyperparameters == {}
