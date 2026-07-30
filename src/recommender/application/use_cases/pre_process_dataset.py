from collections import defaultdict

from recommender.application.dtos.dataset_split import DatasetSplit
from recommender.application.ports.dataset_reader import DatasetReader
from recommender.domain.value_objects.interaction import Interaction
from recommender.domain.value_objects.user_id import UserId


class PreProcessDataset:
    """Caso de uso: lê o dataset e o divide em treino e teste.

    Depende da PORTA DatasetReader (abstração), nunca de pandas/CSV.
    Isso permite testar com um reader falso, em memória.
    """

    def __init__(self, reader: DatasetReader) -> None:
        self._reader = reader

    def execute(self, test_ratio: float = 0.2) -> DatasetSplit:
        """Split temporal POR USUÁRIO (leave-last-out).

        Para cada usuário, as interações mais recentes vão para o teste
        e as mais antigas para o treino. Isso: (1) evita 'ver o futuro'
        de cada usuário e (2) garante que todo usuário de teste também
        aparece no treino — sem cold start artificial na avaliação, que
        é o padrão correto para sistemas de recomendação.

        Usuários com poucas interações ficam inteiros no treino (não há
        como avaliar quem quase não tem histórico).
        """
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must be between 0 and 1")

        by_user: dict[UserId, list[Interaction]] = defaultdict(list)
        for interaction in self._reader.read_interactions():
            by_user[interaction.user_id].append(interaction)

        train: list[Interaction] = []
        test: list[Interaction] = []
        for user_interactions in by_user.values():
            ordered = sorted(user_interactions, key=lambda i: i.occurred_at)
            n_test = int(len(ordered) * test_ratio)
            if n_test == 0:
                train.extend(ordered)  # histórico curto -> tudo no treino
                continue
            cut = len(ordered) - n_test
            train.extend(ordered[:cut])
            test.extend(ordered[cut:])

        return DatasetSplit(train=train, test=test)
