import math


def lambda_schedule(epoch: int, total_epochs: int, lambda_max: float = 1.0, gamma: float = 8.0) -> float:
    """
    Sigmoid warmup schedule for λ (Ganin et al., 2016, Section 5.2.2).

    λ starts at 0 and smoothly ramps to lambda_max over training.
    This lets the encoder learn the task before privacy pressure kicks in.

    Formula: λ_p = lambda_max * (2 / (1 + exp(-γ * p)) - 1)
    where p = epoch / total_epochs ∈ [0, 1]
    """
    p = epoch / total_epochs
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def lr_schedule(epoch: int, total_epochs: int, lr_init: float, alpha: float = 10.0, beta: float = 0.75) -> float:
    """
    Learning rate decay schedule (Ganin et al., 2016, Section 5.2.2).

    Formula: μ_p = μ_0 / (1 + α * p)^β
    where p = epoch / total_epochs ∈ [0, 1]
    """
    p = epoch / total_epochs
    return lr_init / (1.0 + alpha * p) ** beta
