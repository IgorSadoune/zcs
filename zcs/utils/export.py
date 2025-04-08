import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[list] = None,
    output_names: Optional[list] = None,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export.
        save_path: Path to save the ONNX model.
        input_shape: Shape of the input tensor (e.g., (1, 10) for batch_size=1, features=10).
        dynamic_axes: Dynamic axes for variable length inputs/outputs.
        input_names: Names of the input tensors.
        output_names: Names of the output tensors.

    Returns:
        Path to the saved ONNX model.
    """
    try:
        # Check if torch.onnx is available
        import torch.onnx
    except ImportError:
        logger.error(
            "torch.onnx is not available. Make sure you have the right PyTorch version."
        )
        return None

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Set default values
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        # Default to dynamic batch size
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export the model
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,  # Use a widely supported opset version
            do_constant_folding=True,  # Optimize the model
            export_params=True,  # Store the trained parameters
            verbose=False,
        )
        logger.info(f"Model exported to ONNX format at {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        return None


def export_to_torchscript(
    model: torch.nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...],
    method: str = "trace",
) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        model: The PyTorch model to export.
        save_path: Path to save the TorchScript model.
        input_shape: Shape of the input tensor.
        method: Method to use for conversion ('trace' or 'script').

    Returns:
        Path to the saved TorchScript model.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    try:
        if method == "trace":
            # Create dummy input
            dummy_input = torch.randn(input_shape)

            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(save_path)

        elif method == "script":
            # Script the model
            scripted_model = torch.jit.script(model)
            scripted_model.save(save_path)

        else:
            raise ValueError(f"Unsupported method: {method}. Use 'trace' or 'script'.")

        logger.info(f"Model exported to TorchScript format at {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to export model to TorchScript: {e}")
        return None
