"""Treino do modelo de embeddings (PyTorch)."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from recommender.domain.value_objects.hyperparameters import Hyperparameters
from recommender.domain.value_objects.interaction import Interaction
from recommender.infrastructure.models.pytorch_embedding_model import (
    EmbeddingRecommender,
    TorchRecommendationModel,
)
from recommender.infrastructure.preprocessing.movielens_preprocessor import IndexEncoder
from recommender.infrastructure.training.early_stopping import EarlyStopping


class PyTorchTrainer:
    """Implementa ModelTrainer treinando a rede por regressão sobre a nota."""

    def train(
        self,
        interactions: list[Interaction],
        hyperparameters: Hyperparameters,
    ) -> TorchRecommendationModel:
        values = hyperparameters.values
        epochs = int(values.get("epochs", 10))
        learning_rate = float(values.get("learning_rate", 0.01))
        embedding_dim = int(values.get("embedding_dim", 32))
        batch_size = int(values.get("batch_size", 256))
        patience = int(values.get("patience", 3))

        user_encoder: IndexEncoder = IndexEncoder().fit(i.user_id for i in interactions)
        item_encoder: IndexEncoder = IndexEncoder().fit(i.item_id for i in interactions)

        user_index = torch.tensor([user_encoder.index_of(i.user_id) for i in interactions])
        item_index = torch.tensor([item_encoder.index_of(i.item_id) for i in interactions])
        targets = torch.tensor(
            [i.rating.value for i in interactions], dtype=torch.float32
        )

        dataset = TensorDataset(user_index, item_index, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        network = EmbeddingRecommender(
            num_users=len(user_encoder),
            num_items=len(item_encoder),
            embedding_dim=embedding_dim,
        )
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        stopper = EarlyStopping(patience=patience)
        network.train()
        for _epoch in range(epochs):
            running_loss = 0.0
            for batch_users, batch_items, batch_targets in loader:
                optimizer.zero_grad()
                predictions = network(batch_users, batch_items)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_targets)

            epoch_loss = running_loss / len(dataset)
            if stopper.should_stop(epoch_loss):
                break

        return TorchRecommendationModel(network, user_encoder, item_encoder)
