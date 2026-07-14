"""API de inferência (FastAPI).

`create_app` recebe o caso de uso pronto (injeção de dependência) por
isso é testável com um caso de uso falso. `app` (no fim) é a composição
padrão que o uvicorn sobe, lendo caminhos do ambiente.
"""

import os

from fastapi import FastAPI, HTTPException

from recommender.application.dtos.recommendation_output import RecommendationOutput
from recommender.application.use_cases.generate_recommendation import (
    GenerateRecommendation,
)
from recommender.domain.value_objects.user_id import UserId


def create_app(
    generate_recommendation: GenerateRecommendation,
    default_model_name: str = "pytorch",
) -> FastAPI:
    app = FastAPI(title="Recommender API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/users/{user_id}/recommendations")
    def recommend(user_id: int, k: int = 10) -> RecommendationOutput:
        try:
            recommendation = generate_recommendation.execute(
                model_name=default_model_name,
                user_id=UserId(user_id),
                k=k,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return RecommendationOutput.from_domain(recommendation)

    return app


def _build_default_app() -> FastAPI:
    """Composição padrão a partir de variáveis de ambiente.

    Resiliente: se os dados/modelo não estiverem presentes, ainda sobe
    a app (os endpoints é que retornarão erro), para não quebrar no import.
    """
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    model_dir = os.environ.get("MODEL_DIR", "models")
    model_name = os.environ.get("MODEL_NAME", "pytorch")

    from recommender.infrastructure.datasets.movielens_reader import MovieLensReader
    from recommender.infrastructure.datasets.pandas_interaction_repository import (
        PandasInteractionRepository,
    )
    from recommender.infrastructure.persistence.file_pickle_model_repository import (
        FilePickleModelRepository,
    )

    model_repository = FilePickleModelRepository(model_dir)
    try:
        reader = MovieLensReader(
            ratings_path=f"{data_dir}/ratings.csv",
            movies_path=f"{data_dir}/movies.csv",
        )
        interaction_repository = PandasInteractionRepository.from_reader(reader)
    except Exception:  # noqa: BLE001 - dados ausentes não devem impedir o boot
        interaction_repository = PandasInteractionRepository([])

    use_case = GenerateRecommendation(model_repository, interaction_repository)
    return create_app(use_case, model_name)


app = _build_default_app()
