import pickle
from pathlib import Path

from recommender.application.ports.recommendation_model import RecommendationModel


class FilePickleModelRepository:
    """Implementa ModelRepository salvando o modelo em disco (pickle).

    Simples e sem dependências pesadas: o treino salva o .pkl e a API
    o recarrega. Serve de persistência local para o pipeline e o serving.
    (Para versionamento/estágios de produção, use o MlflowModelRegistry.)
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._directory / f"{name}.pkl"

    def save(self, name: str, model: RecommendationModel) -> None:
        with open(self._path(name), "wb") as handle:
            pickle.dump(model, handle)

    def get(self, name: str) -> RecommendationModel | None:
        path = self._path(name)
        if not path.exists():
            return None
        with open(path, "rb") as handle:
            return pickle.load(handle)
