import os
import pickle
import tempfile

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from recommender.application.ports.recommendation_model import RecommendationModel
from recommender.domain.value_objects.user_id import UserId

_PRODUCTION = "Production"


class _RecommenderPyfunc(PythonModel):
    """Empacota um RecommendationModel na interface pyfunc do MLflow.

    Permite salvar/carregar QUALQUER modelo (baseline ou PyTorch) via
    o Model Registry, contanto que ele seja serializável (pickle).
    """

    def load_context(self, context) -> None:  # noqa: ANN001 (assinatura MLflow)
        with open(context.artifacts["model"], "rb") as handle:
            self._model: RecommendationModel = pickle.load(handle)

    def predict(self, context, model_input: pd.DataFrame):  # noqa: ANN001
        results = []
        for _, row in model_input.iterrows():
            recommendation = self._model.recommend(UserId(int(row["user_id"])), int(row["k"]))
            results.append(
                [(item.item_id.value, item.score.value) for item in recommendation.items]
            )
        return results


class MlflowModelRegistry:
    """Adapter: implementa ModelRegistry sobre o MLflow Model Registry.

    NOTA: precisa de um servidor/URI MLflow para funcionar de verdade
    (versionamento e estágios). É exercitado no demo de MLOps / CLI,
    não nos testes unitários do encanamento.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient()

    def register(self, name: str, model: RecommendationModel) -> str:
        model_path = os.path.join(tempfile.mkdtemp(), "model.pkl")
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)

        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=_RecommenderPyfunc(),
                artifacts={"model": model_path},
                registered_model_name=name,
            )

        versions = self._client.search_model_versions(f"name='{name}'")
        latest = max(int(version.version) for version in versions)
        return str(latest)

    def load_production(self, name: str) -> RecommendationModel:
        loaded = mlflow.pyfunc.load_model(f"models:/{name}/{_PRODUCTION}")
        return loaded.unwrap_python_model()._model

    def promote_to_production(self, name: str, version: str) -> None:
        self._client.transition_model_version_stage(
            name=name,
            version=version,
            stage=_PRODUCTION,
            archive_existing_versions=True,
        )
