from recommender.application.dtos.dataset_split import DatasetSplit
from recommender.application.ports.dataset_reader import DatasetReader


class PreProcessDataset:
    """Caso de uso: lê o dataset e o divide em treino e teste.

    Depende da PORTA DatasetReader (abstração), nunca de pandas/CSV.
    Isso permite testar com um reader falso, em memória.
    """

    def __init__(self, reader: DatasetReader) -> None:
        self._reader = reader

    def execute(self, test_ratio: float = 0.2) -> DatasetSplit:
        """Split temporal simples: interações mais antigas -> treino,
        mais recentes -> teste (evita 'ver o futuro' no treino).

        NOTA (para revisitar): um split leave-one-out por usuário é
        mais robusto para recomendação. Deixei o global temporal por
        simplicidade — bom ponto para você melhorar depois.
        """
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must be between 0 and 1")

        interactions = sorted(
            self._reader.read_interactions(),
            key=lambda interaction: interaction.occurred_at,
        )
        cut = int(len(interactions) * (1.0 - test_ratio))
        return DatasetSplit(train=interactions[:cut], test=interactions[cut:])
