import torch
from opacus import PrivacyEngine


def create_private_training(
    model,
    optimizer,
    train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0
):
    """
    Attach Opacus DP-SGD to a client's local model.

    Returns:
        private_model
        private_optimizer
        private_loader
        privacy_engine
    """

    privacy_engine = PrivacyEngine(
        secure_mode=False
    )

    (
        private_model,
        private_optimizer,
        private_loader
    ) = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=True
    )

    return (
        private_model,
        private_optimizer,
        private_loader,
        privacy_engine
    )


def calculate_epsilon(privacy_engine, delta=1e-5):
    """
    Calculate cumulative epsilon for a client.
    """

    return privacy_engine.get_epsilon(delta)