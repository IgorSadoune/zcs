from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer Perceptron (MLP) model.

    This class implements a flexible MLP architecture with configurable
    hidden dimensions, activation functions, dropout, and batch normalization.

    Attributes:
        input_dim (int): Dimension of the input features.
        hidden_dims (List[int]): List of hidden layer dimensions.
        output_dim (int): Dimension of the output layer.
        activation_fn (nn.Module): Activation function to use.
        dropout_rate (float): Dropout probability (0 to disable).
        use_batch_norm (bool): Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation_fn_str: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
    ):
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input features.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimension of the output layer.
            activation_fn_str: Name of the activation function to use.
            dropout_rate: Dropout probability (0 to disable).
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_fn_str = activation_fn_str
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Set activation function
        self.activation_fn = self._get_activation_fn(activation_fn_str)

        # Build the network
        layers = []

        # Input layer
        prev_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (if enabled)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation function
            layers.append(self.activation_fn)

            # Dropout (if enabled)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def _get_activation_fn(self, activation_fn_str: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation_fn_str: Name of the activation function.

        Returns:
            Activation function module.

        Raises:
            ValueError: If the activation function name is not recognized.
        """
        if activation_fn_str.lower() == "relu":
            return nn.ReLU()
        elif activation_fn_str.lower() == "leaky_relu":
            return nn.LeakyReLU(0.1)
        elif activation_fn_str.lower() == "tanh":
            return nn.Tanh()
        elif activation_fn_str.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation_fn_str.lower() == "elu":
            return nn.ELU()
        elif activation_fn_str.lower() == "selu":
            return nn.SELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_str}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            Output tensor of shape [batch_size, output_dim].
        """
        return self.model(x)

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration.

        Returns:
            Dictionary with model configuration.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "activation_fn_str": self.activation_fn_str,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }
