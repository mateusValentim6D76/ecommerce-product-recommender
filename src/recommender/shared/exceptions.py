class RecommenderError(Exception):
    """Exceção base do domínio da aplicação."""


class ModelNotFoundError(RecommenderError):
    """Modelo solicitado não existe no repositório/registro."""


class DatasetNotFoundError(RecommenderError):
    """Arquivo(s) do dataset não encontrado(s)."""
