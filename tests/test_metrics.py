import numpy as np
import pytest
import torch

from zcsearch import MLP, ZeroCostMetrics


@pytest.fixture
def sample_model():
    """Create a sample MLP model for testing."""
    model = MLP(
        input_dim=10, hidden_dims=[32, 32], output_dim=3, activation_fn_str="relu"
    )
    return model


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = torch.randn(20, 10)  # 20 samples, 10 features
    y = torch.randint(0, 3, (20,))  # 3 classes
    return X, y


def test_activation_correlation(sample_model, sample_data):
    """Test activation correlation metric."""
    X, _ = sample_data
    device = torch.device("cpu")

    score = ZeroCostMetrics.compute_activation_correlation(sample_model, X, device)

    assert isinstance(score, float), "Score should be a float"
    assert 0 <= score <= 1, "Score should be between 0 and 1"


def test_grad_conflict(sample_model, sample_data):
    """Test gradient conflict metric."""
    X, y = sample_data
    device = torch.device("cpu")

    score = ZeroCostMetrics.compute_grad_conflict(sample_model, X, y, device, 3)

    assert isinstance(score, float), "Score should be a float"
    assert 0 <= score <= 2, "Score should be between 0 and 2"


def test_zico_score(sample_model, sample_data):
    """Test ZiCo score metric."""
    X, _ = sample_data
    device = torch.device("cpu")

    score = ZeroCostMetrics.compute_zico_score(sample_model, X, device)

    assert isinstance(score, float), "Score should be a float"
    assert score >= 0, "Score should be non-negative"


def test_synflow_score(sample_model, sample_data):
    """Test Synflow score metric."""
    X, _ = sample_data
    device = torch.device("cpu")

    score = ZeroCostMetrics.compute_synflow_score(sample_model, X, device)

    assert isinstance(score, float), "Score should be a float"
    assert score >= 0, "Score should be non-negative"


def test_grasp_score(sample_model, sample_data):
    """Test GraSP score metric."""
    X, y = sample_data
    device = torch.device("cpu")

    score = ZeroCostMetrics.compute_grasp_score(sample_model, X, y, device, 3)

    assert isinstance(score, float), "Score should be a float"
