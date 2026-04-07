import torch
from torch.autograd import Function


class _GradientReverse(Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward pass: identity (does nothing).
    Backward pass: multiplies gradients by -lambda.

    This makes the encoder receive REVERSED gradients from the probe,
    so minimising the probe loss w.r.t. the encoder actually MAXIMISES it.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to a tensor."""
    return _GradientReverse.apply(x, lambda_)
