"""KL autoencoder for forest-proximity vectors with weighted MCR decoding."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from forestgeom import ForestProximity

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError(
        "proximity_ae requires PyTorch; install it with `pip install torch`."
    ) from exc

from mcr import train_decoder


class ProximityAutoencoder(nn.Module):
    """Autoencoder whose softmax output is a distribution over training points."""

    def __init__(self, n_train, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_train, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_train),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class ProximityAEResult:
    """Artifacts from KL training and probability-weighted MCR decoding."""

    encoder: ForestProximity
    autoencoder: ProximityAutoencoder
    latent_train: np.ndarray
    latent_test: np.ndarray
    reconstructed_test_proximity: np.ndarray
    loss_history: list[float]
    X_test_hat: np.ndarray
    device: str
    weight_scheme: str


def reconstruct_proximity_ae(
    leaf_encoder: ForestProximity,
    X_train,
    X_test,
    image_shape,
    *,
    weight_scheme="kerf",
    latent_dim=2,
    epochs=200,
    batch_size=128,
    learning_rate=1e-3,
    random_state=0,
    device="mps",
    n_jobs=1,
):
    """Reconstruct proximity distributions with KL loss and use them as MCR weights."""
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested, but PyTorch MPS is unavailable.")

    proximity_encoder = ForestProximity(
        forest=leaf_encoder.forest_.estimator,
        weight_scheme=weight_scheme,
    ).fit(X_train)
    train_proximity = proximity_encoder.training_proximity(return_dense=False).tocsr()
    test_proximity = proximity_encoder.transform(X_test, return_dense=False).tocsr()
    n_train = train_proximity.shape[0]

    # KeRF rows are stochastic; renormalize only to remove floating-point drift.
    train_sums = np.asarray(train_proximity.sum(axis=1)).ravel()
    test_sums = np.asarray(test_proximity.sum(axis=1)).ravel()
    if np.any(train_sums <= 0) or np.any(test_sums <= 0):
        raise ValueError("KeRF proximity rows must have positive mass.")
    train_proximity = sparse.diags(1.0 / train_sums) @ train_proximity
    test_proximity = sparse.diags(1.0 / test_sums) @ test_proximity

    torch.manual_seed(random_state)
    torch_device = torch.device(device)
    model = ProximityAutoencoder(n_train, latent_dim).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(random_state)

    loss_history = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        permutation = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            indices = permutation[start : start + batch_size]
            batch = torch.from_numpy(
                train_proximity[indices].toarray().astype(np.float32)
            ).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            log_reconstructed = F.log_softmax(model(batch), dim=1)
            loss = F.kl_div(log_reconstructed, batch, reduction="batchmean")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        loss_history.append(total_loss / n_train)

    model.eval()
    latent_train_batches = []
    latent_test_batches = []
    reconstructed_test_batches = []
    with torch.no_grad():
        for start in range(0, n_train, batch_size):
            batch = torch.from_numpy(
                train_proximity[start : start + batch_size]
                .toarray()
                .astype(np.float32)
            ).to(torch_device)
            latent_train_batches.append(model.encoder(batch).cpu().numpy())
        for start in range(0, test_proximity.shape[0], batch_size):
            batch = torch.from_numpy(
                test_proximity[start : start + batch_size]
                .toarray()
                .astype(np.float32)
            ).to(torch_device)
            latent = model.encoder(batch)
            latent_test_batches.append(latent.cpu().numpy())
            reconstructed_test_batches.append(
                torch.softmax(model.decoder(latent), dim=1).cpu().numpy()
            )

    latent_train = np.vstack(latent_train_batches)
    latent_test = np.vstack(latent_test_batches)
    reconstructed_test = np.vstack(reconstructed_test_batches)

    # Decode every training leaf box once, then probability-average those MCRs.
    columns = list(range(X_train.shape[1]))
    metadata = pd.DataFrame(
        {
            "variable": columns,
            "class": "numeric",
            "decimals": 0,
            "min": np.min(X_train, axis=0),
            "max": np.max(X_train, axis=0),
        }
    )
    emap = SimpleNamespace(
        leafIDs=leaf_encoder.cache_.leaf_matrix,
        meta={"metadata": metadata, "input_class": ["np.ndarray"]},
    )
    rng_state = np.random.get_state()
    np.random.seed(random_state)
    try:
        training_mcr = train_decoder(
            leaf_encoder.forest_.estimator, emap, n_jobs=n_jobs
        ).to_numpy()
    finally:
        np.random.set_state(rng_state)
    decoded = reconstructed_test @ training_mcr

    return ProximityAEResult(
        encoder=proximity_encoder,
        autoencoder=model,
        latent_train=latent_train,
        latent_test=latent_test,
        reconstructed_test_proximity=reconstructed_test,
        loss_history=loss_history,
        X_test_hat=decoded.reshape(-1, *image_shape),
        device=str(torch_device),
        weight_scheme=weight_scheme,
    )
