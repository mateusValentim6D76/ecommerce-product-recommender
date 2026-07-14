"""[VOCÊ IMPLEMENTA] O loop de treino PyTorch.

Este é o coração de ML do projeto. O trainer recebe as interações e os
hiperparâmetros e devolve um modelo treinado (TorchRecommendationModel).

Você preenche o método `train`. A estrutura e as dependências já estão
montadas — foque na lógica de ML.
"""

import torch

from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.infrastructure.models.pytorch_embedding_model import (
    EmbeddingRecommender,
    TorchRecommendationModel,
)
from recommender.infrastructure.preprocessing.movielens_preprocessor import IndexEncoder
from recommender.infrastructure.training.early_stopping import EarlyStopping


class PyTorchTrainer:
    """Satisfaz a porta ModelTrainer."""

    def train(
        self,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> TorchRecommendationModel:
        # PASSO A PASSO sugerido (cada linha vira algumas de código):
        #
        # 1. HIPERPARÂMETROS: leia com defaults, ex.:
        #    epochs = int(hyperparameters.values.get("epochs", 10))
        #    lr = hyperparameters.values.get("learning_rate", 0.01)
        #    embedding_dim = int(hyperparameters.values.get("embedding_dim", 32))
        #
        # 2. ENCODERS: mapeie ids -> índices densos (embedding precisa disso)
        #    user_encoder = IndexEncoder().fit(i.user_id for i in interactions)
        #    item_encoder = IndexEncoder().fit(i.item_id for i in interactions)
        #
        # 3. TENSORES: monte tensores de user_idx, item_idx e target (nota).
        #    Use torch.tensor(...). Considere uma FeedbackStrategy p/ o alvo.
        #
        # 4. MODELO + OTIMIZADOR + LOSS:
        #    net = EmbeddingRecommender(len(user_encoder), len(item_encoder), embedding_dim)
        #    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        #    loss_fn = torch.nn.MSELoss()   # regressão sobre a nota
        #
        # 5. LOOP DE TREINO (por época): forward -> loss -> backward -> step.
        #    Use o EarlyStopping (abaixo) monitorando a loss:
        #    stopper = EarlyStopping(patience=3)
        #    ... if stopper.should_stop(epoch_loss): break
        #
        # 6. Devolva TorchRecommendationModel(net, user_encoder, item_encoder).
        raise NotImplementedError("Implemente o loop de treino PyTorch")
