# zero_cost_search/zero_cost_search.py

"""
Zero-Cost Neural Architecture Search for MLPs

This module implements efficient zero-cost proxies for neural architecture search,
incorporating recent developments that build upon Mellor et al.'s approach.

Key features:
1. Multiple zero-cost proxies from recent literature
2. Ensemble scoring for more reliable architecture evaluation
3. Lightweight meta-learning for dataset-aware architecture prediction

References:
- "Neural Architecture Search without Training" (Mellor et al., 2021)
- "Zero-Cost Proxies for Lightweight NAS" (Abdelfattah et al., 2021)
- "ZiCo: Zero-shot NAS via Inverse Coefficient of Variation on Gradients" (Li et al., 2022)
- "Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets" (Yan et al., 2022)
"""

import logging
import os
import pickle
import random
import time
from collections import defaultdict
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .models import MLP

# Configure logging
logger = logging.getLogger(__name__)


class ZeroCostMetrics:
    """Collection of zero-cost metrics for evaluating neural network architectures.

    This class implements various metrics from recent literature that can evaluate
    neural network architectures without training.
    """

    @staticmethod
    def compute_activation_correlation(
        model: nn.Module, X: torch.Tensor, device: torch.device
    ) -> float:
        """Compute activation correlation score (Mellor et al., 2021).

        Args:
            model: The model to evaluate
            X: Input data tensor
            device: Device to run computation on

        Returns:
            Correlation score
        """
        batch_size = min(100, X.size(0))
        X_sample = X[:batch_size].to(device)

        # Register hooks to collect activations
        activations = []
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                activations.append(output.detach())

        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(X_sample)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not activations:
            return 0.0

        # Compute correlation for each layer
        layer_scores = []
        for acts in activations:
            # Reshape activations: [batch_size, ...] -> [batch_size, -1]
            flat_acts = acts.view(batch_size, -1)

            # Normalize each sample's activations
            acts_mean = flat_acts.mean(dim=1, keepdim=True)
            acts_std = flat_acts.std(dim=1, keepdim=True) + 1e-8
            flat_acts_norm = (flat_acts - acts_mean) / acts_std

            # Compute pairwise correlations between samples
            corr_matrix = torch.mm(
                flat_acts_norm, flat_acts_norm.t()
            ) / flat_acts_norm.size(1)

            # Average the off-diagonal elements
            mask = torch.ones_like(corr_matrix) - torch.eye(batch_size, device=device)
            mean_corr = (corr_matrix * mask).sum() / (batch_size * (batch_size - 1))

            layer_scores.append(mean_corr.item())

        return np.mean(layer_scores)

    @staticmethod
    def compute_grad_conflict(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        num_classes: int,
    ) -> float:
        """Compute gradient conflict score (Abdelfattah et al., 2021).

        Lower gradient conflict indicates better architecture.

        Args:
            model: The model to evaluate
            X: Input data tensor
            y: Target data tensor
            device: Device to run computation on
            num_classes: Number of output classes

        Returns:
            Gradient conflict score (lower is better)
        """
        batch_size = min(32, X.size(0))
        X_sample = X[:batch_size].clone().to(device)
        X_sample.requires_grad_(True)

        # For classification
        if num_classes > 1:
            y_sample = y[:batch_size].to(device)
            criterion = nn.CrossEntropyLoss(reduction="none")
        else:  # For regression
            y_sample = y[:batch_size].to(device)
            criterion = nn.MSELoss(reduction="none")

        model.zero_grad()
        model.train()  # Important for computing gradients

        # Forward pass
        outputs = model(X_sample)

        # Compute per-sample gradients
        sample_grads = []
        for i in range(batch_size):
            model.zero_grad()
            if X_sample.grad is not None:
                X_sample.grad.zero_()

            if num_classes > 1:
                loss = criterion(outputs[i : i + 1], y_sample[i : i + 1])
            else:
                loss = criterion(outputs[i : i + 1].squeeze(), y_sample[i : i + 1])

            loss.backward(retain_graph=(i < batch_size - 1))

            # Get gradients for all parameters
            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1).detach().clone())

            if grads:
                sample_grads.append(torch.cat(grads))

        if not sample_grads or len(sample_grads) < 2:
            return 0.0

        # Compute pairwise cosine similarities between gradients
        conflict_score = 0.0
        count = 0
        for i in range(len(sample_grads)):
            for j in range(i + 1, len(sample_grads)):
                cos_sim = F.cosine_similarity(
                    sample_grads[i].unsqueeze(0), sample_grads[j].unsqueeze(0)
                )
                # Convert similarity to conflict (1 - similarity)
                conflict_score += 1.0 - cos_sim.item()
                count += 1

        if count == 0:
            return 0.0

        return conflict_score / count

    @staticmethod
    def compute_zico_score(
        model: nn.Module, X: torch.Tensor, device: torch.device
    ) -> float:
        """Compute ZiCo score (Li et al., 2022).

        ZiCo measures the inverse coefficient of variation of gradients.
        Higher ZiCo score indicates better architecture.

        Args:
            model: The model to evaluate
            X: Input data tensor
            device: Device to run computation on

        Returns:
            ZiCo score
        """
        batch_size = min(64, X.size(0))
        X_sample = X[:batch_size].to(device)
        X_sample.requires_grad_(True)

        model.zero_grad()
        model.eval()  # Still works in eval mode for this metric

        # Forward pass
        outputs = model(X_sample)

        # Use sum of outputs as a simple objective function
        loss = outputs.sum()
        loss.backward()

        # Compute ZiCo for each parameter
        zico_scores = []
        for name, param in model.named_parameters():
            if param.grad is not None and "bias" not in name:  # Skip bias terms
                grad = param.grad.detach()

                # Compute mean and standard deviation
                mean = grad.abs().mean()
                std = grad.abs().std()

                if mean > 0 and std > 0:
                    # Inverse coefficient of variation: mean / std
                    zico = mean / (std + 1e-10)
                    zico_scores.append(zico.item())

        if not zico_scores:
            return 0.0

        return np.mean(zico_scores)

    @staticmethod
    def compute_synflow_score(
        model: nn.Module, X: torch.Tensor, device: torch.device
    ) -> float:
        """Compute Synflow score (Tanaka et al., 2020).

        Synflow measures the product of parameters and gradients.
        Higher Synflow score indicates better architecture.

        Args:
            model: The model to evaluate
            X: Input data tensor
            device: Device to run computation on

        Returns:
            Synflow score
        """
        # Save original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            # Set all parameters to 1 (as per Synflow method)
            param.data = torch.ones_like(param.data)

        # Create a dummy input with same shape but all ones
        dummy_input = torch.ones_like(X[:1]).to(device)
        dummy_input.requires_grad_(True)

        model.zero_grad()
        model.eval()

        # Forward pass
        output = model(dummy_input)

        # Use sum of absolute values as the objective function
        torch.sum(torch.abs(output)).backward()

        # Compute Synflow score
        synflow = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                synflow += (param * param.grad).sum().item()

        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]

        return synflow

    @staticmethod
    def compute_grasp_score(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        num_classes: int,
    ) -> float:
        """Compute GraSP score (Wang et al., 2020).

        GraSP measures the gradient signal preservation.
        Higher GraSP score indicates better architecture.

        Args:
            model: The model to evaluate
            X: Input data tensor
            y: Target data tensor
            device: Device to run computation on
            num_classes: Number of output classes

        Returns:
            GraSP score
        """
        batch_size = min(64, X.size(0))
        X_sample = X[:batch_size].to(device)

        # For classification
        if num_classes > 1:
            y_sample = y[:batch_size].to(device)
            criterion = nn.CrossEntropyLoss()
        else:  # For regression
            y_sample = y[:batch_size].to(device)
            criterion = nn.MSELoss()

        # First backward pass to compute gradients
        model.zero_grad()
        model.train()

        outputs = model(X_sample)
        if num_classes > 1:
            loss = criterion(outputs, y_sample)
        else:
            loss = criterion(outputs.squeeze(), y_sample)

        loss.backward(create_graph=True)

        # Store first-order gradients
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()

        # Second backward pass to compute Hessian-gradient product
        model.zero_grad()
        grasp_score = 0.0

        for name, param in model.named_parameters():
            if name in grads:
                # Compute gradient of the gradient
                torch.sum(grads[name]).backward(retain_graph=True)

                # GraSP score is the negative of the product of gradient and Hessian-gradient
                if param.grad is not None:
                    grasp_score -= (grads[name] * param.grad).sum().item()

                model.zero_grad()

        return grasp_score


class DatasetFeatureExtractor:
    """Extract features from datasets for meta-learning.

    This class computes statistical features from datasets that can be used
    to predict good architectures without extensive search.
    """

    @staticmethod
    def extract_features(
        X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Extract statistical features from a dataset.

        Args:
            X: Input data tensor
            y: Optional target data tensor

        Returns:
            Dictionary of dataset features
        """
        # Convert to numpy for easier statistical analysis
        X_np = X.cpu().numpy()

        features = {}

        # Basic statistics
        features["input_dim"] = X.shape[1]
        features["num_samples"] = X.shape[0]

        # Data statistics
        features["mean_abs"] = np.mean(np.abs(X_np))
        features["std"] = np.std(X_np)
        features["skewness"] = np.mean(
            ((X_np - np.mean(X_np)) / (np.std(X_np) + 1e-8)) ** 3
        )
        features["kurtosis"] = (
            np.mean(((X_np - np.mean(X_np)) / (np.std(X_np) + 1e-8)) ** 4) - 3
        )

        # Correlation structure
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X_np.T)
            features["mean_correlation"] = np.mean(
                np.abs(corr_matrix - np.eye(X.shape[1]))
            )
        else:
            features["mean_correlation"] = 0.0

        # PCA to measure intrinsic dimensionality
        if X.shape[0] > 10 and X.shape[1] > 1:
            try:
                # Standardize data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_np)

                # Apply PCA
                pca = PCA()
                pca.fit(X_scaled)

                # Compute effective dimension (number of components to explain 90% variance)
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                effective_dim = np.argmax(cumulative_variance >= 0.9) + 1
                features["effective_dim"] = (
                    effective_dim / X.shape[1]
                )  # Normalize by input dimension

                # First principal component variance ratio
                features["first_pc_var_ratio"] = explained_variance_ratio[0]
            except:
                features["effective_dim"] = 1.0
                features["first_pc_var_ratio"] = 1.0
        else:
            features["effective_dim"] = 1.0
            features["first_pc_var_ratio"] = 1.0

        # Target statistics (if provided)
        if y is not None:
            y_np = y.cpu().numpy()

            # Check if classification or regression
            if len(y.unique()) < 20:  # Likely classification
                features["task_type"] = 0.0  # Classification
                features["num_classes"] = len(y.unique())

                # Class balance
                class_counts = np.bincount(y_np.astype(int))
                class_probs = class_counts / len(y_np)
                features["class_entropy"] = -np.sum(
                    class_probs * np.log(class_probs + 1e-10)
                )
                features["class_imbalance"] = np.max(class_probs) - np.min(class_probs)
            else:  # Likely regression
                features["task_type"] = 1.0  # Regression
                features["num_classes"] = 1.0
                features["target_mean"] = np.mean(y_np)
                features["target_std"] = np.std(y_np)
                features["target_skewness"] = np.mean(
                    ((y_np - np.mean(y_np)) / (np.std(y_np) + 1e-8)) ** 3
                )

        return features


class ArchitecturePredictor:
    """Predict good architectures based on dataset features.

    This class implements a lightweight meta-learning approach to predict
    good architectures directly from dataset features.
    """

    def __init__(self):
        """Initialize the architecture predictor."""
        # Rules derived from empirical observations across datasets
        self.rules = [
            # Rule format: (condition_fn, architecture_fn)
            # Each condition_fn takes dataset features and returns True/False
            # Each architecture_fn takes dataset features and returns architecture config
            # Rule 1: High-dimensional data with low correlation -> wider networks
            (
                lambda f: f["input_dim"] > 50 and f["mean_correlation"] < 0.3,
                lambda f: {"hidden_dims": [256, 256], "activation_fn_str": "relu"},
            ),
            # Rule 2: High intrinsic dimensionality -> deeper networks
            (
                lambda f: f["effective_dim"] > 0.5,
                lambda f: {"hidden_dims": [128, 128, 128], "activation_fn_str": "relu"},
            ),
            # Rule 3: Highly non-linear data (high kurtosis) -> tanh activation
            (
                lambda f: f["kurtosis"] > 5.0,
                lambda f: {"hidden_dims": [128, 128], "activation_fn_str": "tanh"},
            ),
            # Rule 4: Many classes with balanced distribution -> pyramid architecture
            (
                lambda f: f.get("task_type", 0) == 0
                and f.get("num_classes", 0) > 5
                and f.get("class_entropy", 0) > 0.9,
                lambda f: {"hidden_dims": [256, 128, 64], "activation_fn_str": "relu"},
            ),
            # Rule 5: Regression with high variance -> leaky_relu
            (
                lambda f: f.get("task_type", 0) == 1 and f.get("target_std", 0) > 1.0,
                lambda f: {
                    "hidden_dims": [128, 128],
                    "activation_fn_str": "leaky_relu",
                },
            ),
            # Rule 6: Small dataset -> smaller network
            (
                lambda f: f["num_samples"] < 1000,
                lambda f: {"hidden_dims": [64, 64], "activation_fn_str": "relu"},
            ),
            # Default rule
            (
                lambda f: True,
                lambda f: {"hidden_dims": [128, 128], "activation_fn_str": "relu"},
            ),
        ]

    def predict_architecture(
        self, dataset_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict a good architecture based on dataset features.

        Args:
            dataset_features: Features extracted from the dataset

        Returns:
            Dictionary with architecture configuration
        """
        # Apply the first matching rule
        for condition_fn, architecture_fn in self.rules:
            if condition_fn(dataset_features):
                return architecture_fn(dataset_features)

        # Fallback (should never reach here due to default rule)
        return {"hidden_dims": [128, 128], "activation_fn_str": "relu"}


class ZeroCostNAS:
    """Zero-Cost Neural Architecture Search for MLPs.

    This class implements an efficient approach to neural architecture search
    that combines multiple zero-cost proxies and meta-learning.

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output layer.
        device (torch.device): Device to run the models on.
        seed (int): Random seed for reproducibility.
        best_config (Dict[str, Any]): Configuration of the best performing model.
        best_score (float): Score of the best performing model.
        best_model_state (Dict[str, torch.Tensor]): State dict of the best model.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: Optional[torch.device] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """Initialize the ZeroCostNAS.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output layer.
            device: Device to run the models on. If None, uses CUDA if available, else CPU.
            seed: Random seed for reproducibility.
            cache_dir: Directory to cache results. If None, no caching is used.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.cache_dir = cache_dir

        self.best_config = None
        self.best_score = float("-inf")
        self.best_model_state = None

        self.metrics = ZeroCostMetrics()
        self.feature_extractor = DatasetFeatureExtractor()
        self.architecture_predictor = ArchitecturePredictor()

        # Set up logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")

        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info(
            f"Initialized ZeroCostNAS with input_dim={input_dim}, output_dim={output_dim}"
        )
        logger.info(f"Using device: {self.device}")

        # Set seeds for reproducibility
        self._set_seeds(seed)

        # Create cache directory if needed
        if cache_dir is not None and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: The random seed to use.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"Set random seed to {seed}")

    def _create_model(self, config: Dict[str, Any]) -> MLP:
        """Create an MLP model with the given configuration.

        Args:
            config: Dictionary containing model configuration parameters.

        Returns:
            An initialized MLP model.
        """
        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=config["hidden_dims"],
            output_dim=self.output_dim,
            activation_fn_str=config["activation_fn_str"],
            dropout_rate=0.0,  # No dropout for evaluation
            use_batch_norm=False,  # No batch norm for evaluation
        )
        model.to(self.device)
        return model

    def _compute_ensemble_score(
        self, model: nn.Module, X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute an ensemble score using multiple zero-cost proxies.

        Args:
            model: The model to evaluate
            X: Input data tensor
            y: Optional target data tensor

        Returns:
            Dictionary with individual metric scores and ensemble score
        """
        scores = {}

        # Compute activation correlation (Mellor et al.)
        scores["correlation"] = ZeroCostMetrics.compute_activation_correlation(
            model, X, self.device
        )
        logger.debug(f"Correlation score: {scores['correlation']:.4f}")

        # Compute ZiCo score (Li et al.)
        scores["zico"] = ZeroCostMetrics.compute_zico_score(model, X, self.device)
        logger.debug(f"ZiCo score: {scores['zico']:.4f}")

        # Compute Synflow score (Tanaka et al.)
        scores["synflow"] = ZeroCostMetrics.compute_synflow_score(model, X, self.device)
        logger.debug(f"Synflow score: {scores['synflow']:.4f}")

        # Compute gradient conflict if targets are provided (Abdelfattah et al.)
        if y is not None:
            scores["grad_conflict"] = (
                -ZeroCostMetrics.compute_grad_conflict(  # Negate because lower is better
                    model, X, y, self.device, self.output_dim
                )
            )
            logger.debug(f"Gradient conflict score: {scores['grad_conflict']:.4f}")

            # Compute GraSP score (Wang et al.)
            scores["grasp"] = ZeroCostMetrics.compute_grasp_score(
                model, X, y, self.device, self.output_dim
            )
            logger.debug(f"GraSP score: {scores['grasp']:.4f}")

        # Normalize scores to [0, 1] range
        max_scores = {
            "correlation": 1.0,
            "zico": 10.0,
            "synflow": 1e6,
            "grad_conflict": 0.0,
            "grasp": 1e6,
        }
        min_scores = {
            "correlation": 0.0,
            "zico": 0.0,
            "synflow": 0.0,
            "grad_conflict": -2.0,
            "grasp": -1e6,
        }

        normalized_scores = {}
        for metric, score in scores.items():
            normalized_scores[metric] = (score - min_scores[metric]) / (
                max_scores[metric] - min_scores[metric]
            )
            normalized_scores[metric] = max(
                0.0, min(1.0, normalized_scores[metric])
            )  # Clip to [0, 1]

        # Weighted ensemble
        weights = {
            "correlation": 0.3,
            "zico": 0.2,
            "synflow": 0.2,
            "grad_conflict": 0.15,
            "grasp": 0.15,
        }

        ensemble_score = 0.0
        weight_sum = 0.0

        for metric, weight in weights.items():
            if metric in normalized_scores:
                ensemble_score += weight * normalized_scores[metric]
                weight_sum += weight

        if weight_sum > 0:
            ensemble_score /= weight_sum

        # Add ensemble score to the results
        scores["ensemble"] = ensemble_score
        logger.info(f"Ensemble score: {ensemble_score:.4f}")

        return scores

    def _get_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate a cache key for a configuration.

        Args:
            config: Model configuration

        Returns:
            Cache key string
        """
        key_parts = [
            f"dims_{'-'.join(map(str, config['hidden_dims']))}",
            f"act_{config['activation_fn_str']}",
            f"seed_{self.seed}",
        ]
        return "_".join(key_parts)

    def _get_cached_score(self, config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get a cached score for a configuration if available.

        Args:
            config: Model configuration

        Returns:
            Cached scores or None if not found
        """
        if self.cache_dir is None:
            return None

        cache_key = self._get_cache_key(config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                return cached_data.get("scores")
            except:
                logger.warning(f"Failed to load cache file: {cache_file}")
                return None

        return None

    def _cache_score(
        self,
        config: Dict[str, Any],
        scores: Dict[str, float],
        model_state: Dict[str, torch.Tensor],
    ) -> None:
        """Cache scores and model state for a configuration.

        Args:
            config: Model configuration
            scores: Computed scores
            model_state: Model state dictionary
        """
        if self.cache_dir is None:
            return

        cache_key = self._get_cache_key(config)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        cached_data = {
            "config": config,
            "scores": scores,
            "model_state": {k: v.cpu() for k, v in model_state.items()},
        }

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            logger.debug(f"Cached results to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def predict_architecture(
        self, X: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Predict a good architecture directly from dataset features.

        This method uses meta-learning to predict a good architecture without
        evaluating any models.

        Args:
            X: Input data tensor
            y: Optional target data tensor

        Returns:
            Dictionary with predicted architecture configuration
        """
        # Extract dataset features
        dataset_features = DatasetFeatureExtractor.extract_features(X, y)
        logger.info("Extracted dataset features for architecture prediction")

        # Predict architecture
        architecture = self.architecture_predictor.predict_architecture(
            dataset_features
        )
        logger.info(f"Predicted architecture: {architecture}")

        return architecture

    def search(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        depths: List[int] = [2, 3, 4],
        widths: List[int] = [64, 128, 256],
        activations: List[str] = ["relu", "tanh", "leaky_relu"],
        num_samples: int = 20,
        use_meta_learning: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Search for the best architecture using zero-cost proxies.

        Args:
            X: Input data tensor
            y: Optional target data tensor
            depths: List of network depths to test
            widths: List of layer widths to test
            activations: List of activation functions to test
            num_samples: Number of random samples to draw from the configuration space
            use_meta_learning: Whether to use meta-learning to guide the search
            verbose: Whether to print progress information

        Returns:
            Dictionary containing the best configuration, scores, and search time
        """
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)

        # Reset best score and config
        self.best_score = float("-inf")
        self.best_config = None
        self.best_model_state = None
        self.all_results = []

        start_time = time.time()
        logger.info("Starting architecture search")

        # If using meta-learning, start with the predicted architecture
        if use_meta_learning:
            predicted_config = self.predict_architecture(X, y)
            if verbose:
                print(f"Meta-learning predicted architecture: {predicted_config}")
            logger.info(f"Meta-learning predicted architecture: {predicted_config}")

            # Add the predicted architecture to the search space
            all_configs = [predicted_config]

            # Generate additional configurations around the predicted one
            hidden_dims = predicted_config["hidden_dims"]
            activation = predicted_config["activation_fn_str"]

            # Variations of the predicted architecture
            for scale in [0.5, 0.75, 1.25, 1.5]:
                scaled_dims = [max(16, int(dim * scale)) for dim in hidden_dims]
                all_configs.append(
                    {"hidden_dims": scaled_dims, "activation_fn_str": activation}
                )

            # Add some with different activations
            for act in activations:
                if act != activation:
                    all_configs.append(
                        {"hidden_dims": hidden_dims, "activation_fn_str": act}
                    )

            # Add some with different depths
            for depth in depths:
                if depth != len(hidden_dims):
                    if depth < len(hidden_dims):
                        new_dims = hidden_dims[:depth]
                    else:
                        new_dims = hidden_dims + [hidden_dims[-1]] * (
                            depth - len(hidden_dims)
                        )
                    all_configs.append(
                        {"hidden_dims": new_dims, "activation_fn_str": activation}
                    )
        else:
            # Generate all possible hidden dimension configurations
            hidden_dims_configs = []
            for depth in depths:
                for width in widths:
                    # Create hidden dimensions list based on depth and width
                    if isinstance(width, int):
                        # Constant width across all layers
                        hidden_dims_configs.append([width] * depth)
                    else:
                        # Variable width across layers
                        hidden_dims_configs.append(
                            width[:depth]
                            if len(width) >= depth
                            else width + [width[-1]] * (depth - len(width))
                        )

            # Generate all configurations
            all_configs = []
            for hidden_dims, activation in product(hidden_dims_configs, activations):
                config = {"hidden_dims": hidden_dims, "activation_fn_str": activation}
                all_configs.append(config)

        # If there are too many configurations, sample randomly
        if len(all_configs) > num_samples:
            random.seed(self.seed)
            all_configs = random.sample(all_configs, num_samples)

        if verbose:
            print(
                f"Testing {len(all_configs)} configurations using zero-cost proxies..."
            )
        logger.info(
            f"Testing {len(all_configs)} configurations using zero-cost proxies"
        )

        # Test each configuration
        for i, config in enumerate(all_configs):
            # Check cache first
            cached_scores = self._get_cached_score(config)

            if cached_scores is not None:
                scores = cached_scores
                if verbose:
                    print(f"Configuration {i+1}/{len(all_configs)}: {config} (cached)")
                    print(f"Ensemble score: {scores['ensemble']:.4f}")
                logger.info(
                    f"Using cached scores for configuration {i+1}/{len(all_configs)}: {config}"
                )
            else:
                # Set seed for reproducibility
                self._set_seeds(self.seed)

                # Create and initialize the model
                model = self._create_model(config)

                # Compute the ensemble score
                scores = self._compute_ensemble_score(model, X, y)

                # Cache the result
                self._cache_score(config, scores, model.state_dict())

                if verbose:
                    print(f"Configuration {i+1}/{len(all_configs)}: {config}")
                    print(f"Ensemble score: {scores['ensemble']:.4f}")
                logger.info(
                    f"Evaluated configuration {i+1}/{len(all_configs)}: {config}"
                )

            # Store all results
            self.all_results.append({"config": config, "scores": scores})

            # Update best configuration if needed
            if scores["ensemble"] > self.best_score:
                self.best_score = scores["ensemble"]
                self.best_config = config.copy()
                self.best_scores = scores.copy()

                # If not from cache, save the model state
                if cached_scores is None:
                    self.best_model_state = model.state_dict()

                if verbose:
                    print(
                        f"New best configuration found! Score: {scores['ensemble']:.4f}"
                    )
                logger.info(
                    f"New best configuration found! Score: {scores['ensemble']:.4f}"
                )

        end_time = time.time()
        search_time = end_time - start_time

        if verbose:
            print(f"\nSearch completed in {search_time:.2f} seconds")
            print("Best configuration:")
            print(self.best_config)
            print(f"Best score: {self.best_score:.4f}")
        logger.info(f"Search completed in {search_time:.2f} seconds")
        logger.info(f"Best configuration: {self.best_config}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return {
            "best_config": self.best_config,
            "best_scores": self.best_scores,
            "best_score": self.best_score,
            "search_time": search_time,
            "all_results": self.all_results,
        }

    def get_best_model(
        self, include_dropout: float = 0.0, include_batchnorm: bool = False
    ) -> MLP:
        """Get the best model with its weights.

        Args:
            include_dropout: Dropout rate to use in the final model.
            include_batchnorm: Whether to include batch normalization in the final model.

        Returns:
            The best performing MLP model.
        """
        if self.best_config is None:
            raise ValueError("No search has been performed yet. Call search first.")

        # Set seed for reproducibility
        self._set_seeds(self.seed)

        # Create the model with the best configuration, but potentially with dropout and batchnorm
        config = self.best_config.copy()

        model = MLP(
            input_dim=self.input_dim,
            hidden_dims=config["hidden_dims"],
            output_dim=self.output_dim,
            activation_fn_str=config["activation_fn_str"],
            dropout_rate=include_dropout,
            use_batch_norm=include_batchnorm,
        )
        model.to(self.device)

        # Load the saved state for the layers that match
        if self.best_model_state is not None:
            # Filter the state dict to only include keys that exist in the new model
            filtered_state = {
                k: v
                for k, v in self.best_model_state.items()
                if k in model.state_dict() and v.shape == model.state_dict()[k].shape
            }
            model.load_state_dict(filtered_state, strict=False)
            logger.info("Loaded best model state")

        return model

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of the search results.

        Returns:
            Dictionary with search results summary
        """
        if not hasattr(self, "all_results") or not self.all_results:
            raise ValueError("No search has been performed yet. Call search first.")

        # Sort configurations by score
        sorted_results = sorted(
            self.all_results, key=lambda x: x["scores"]["ensemble"], reverse=True
        )

        # Extract top configurations
        top_configs = []
        for result in sorted_results[:5]:  # Top 5 configurations
            top_configs.append(
                {
                    "config": result["config"],
                    "ensemble_score": result["scores"]["ensemble"],
                    "individual_scores": {
                        k: v for k, v in result["scores"].items() if k != "ensemble"
                    },
                }
            )

        # Compute statistics across all configurations
        all_scores = [result["scores"]["ensemble"] for result in self.all_results]
        score_stats = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "min": np.min(all_scores),
            "max": np.max(all_scores),
            "median": np.median(all_scores),
        }

        # Analyze impact of different architectural choices
        depth_scores = defaultdict(list)
        width_scores = defaultdict(list)
        activation_scores = defaultdict(list)

        for result in self.all_results:
            config = result["config"]
            score = result["scores"]["ensemble"]

            depth_scores[len(config["hidden_dims"])].append(score)
            width_scores[config["hidden_dims"][0]].append(
                score
            )  # Use first layer width as proxy
            activation_scores[config["activation_fn_str"]].append(score)

        # Compute average score for each architectural choice
        depth_analysis = {
            depth: np.mean(scores) for depth, scores in depth_scores.items()
        }
        width_analysis = {
            width: np.mean(scores) for width, scores in width_scores.items()
        }
        activation_analysis = {
            act: np.mean(scores) for act, scores in activation_scores.items()
        }

        return {
            "top_configurations": top_configs,
            "score_statistics": score_stats,
            "depth_analysis": depth_analysis,
            "width_analysis": width_analysis,
            "activation_analysis": activation_analysis,
            "best_config": self.best_config,
            "best_score": self.best_score,
        }
