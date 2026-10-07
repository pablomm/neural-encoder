"""Residual networks for refining existing representations."""

import torch
from torch import nn

__all__ = ["ResidualNetwork", "ResidualMLP"]


class ResidualNetwork(nn.Module):
    """Wrap a residual branch as f(z, y) = s * (z + alpha * network(y)).

    Parameters
    ----------
    network : torch.nn.Module
        Residual branch whose output matches the base representation shape.
    initial_alpha : float, default=0.25
        Initial residual multiplier.
    trainable_alpha : bool, default=True
        Whether to optimize alpha. Otherwise it is a registered buffer.
    output_scale : torch.Tensor or None, default=None
        Optional per-feature multiplier s applied to the output, stored as
        the buffer ``output_scale``. None leaves the output unscaled (s = 1).
        NonlinearRefiner sets it after training when output reweighting is
        enabled; pass a tensor of the right shape to load such a state dict.
    """

    def __init__(
        self, network: nn.Module, *, initial_alpha: float = 0.25,
        trainable_alpha: bool = True, output_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        alpha = torch.tensor(float(initial_alpha))
        if trainable_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)
        self.register_buffer("output_scale", None if output_scale is None else torch.as_tensor(output_scale))

    def forward(self, x: torch.Tensor, network_input: torch.Tensor | None = None) -> torch.Tensor:
        """Refine x using network_input, or x itself when omitted."""
        residual = self.forward_residual(x if network_input is None else network_input)
        if residual.shape != x.shape:
            raise ValueError("The residual output must match the base representation shape.")
        output = x + self.alpha * residual
        return output if self.output_scale is None else output * self.output_scale

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual branch output before alpha scaling."""
        return self.network(x)


class ResidualMLP(ResidualNetwork):
    """Residual MLP with GELU, dropout, and an initially zero residual.

    n_features is the output dimension. network_input_features defaults to
    n_features and specifies the residual branch input dimension.
    depth counts hidden layers. The final linear layer is initialized to
    zero, so the initial network is exactly the identity. Other linear
    layers use orthogonal weights and zero biases. Use a nonzero initial
    alpha to allow gradients through the initially zero residual branch.
    """

    def __init__(
        self, n_features: int, *, network_input_features: int | None = None, hidden_dim: int = 768, depth: int = 1,
        dropout: float = 0.1, initial_alpha: float = 0.25,
        trainable_alpha: bool = True, output_scale: torch.Tensor | None = None,
    ) -> None:
        if n_features < 1 or hidden_dim < 1 or depth < 1:
            raise ValueError("n_features, hidden_dim, and depth must be positive.")
        width = n_features if network_input_features is None else network_input_features
        if width < 1:
            raise ValueError("network_input_features must be positive.")
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            width = hidden_dim
        layers.append(nn.Linear(width, n_features))
        network = nn.Sequential(*layers)
        for layer in network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(network[-1].weight)
        super().__init__(network, initial_alpha=initial_alpha, trainable_alpha=trainable_alpha,
                         output_scale=output_scale)
