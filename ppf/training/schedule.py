import math


def lambda_schedule(epoch: int, total_epochs: int, lambda_max: float = 1.0, gamma: float = 8.0) -> float:
    p = epoch / total_epochs
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def lr_schedule(epoch: int, total_epochs: int, lr_init: float, alpha: float = 10.0, beta: float = 0.75) -> float:
    p = epoch / total_epochs
    return lr_init / (1.0 + alpha * p) ** beta
