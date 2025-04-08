import os
import tempfile

import pytest
import torch

from zcsearch import ZeroCostNAS


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = torch.randn(20, 10)  # 20 samples, 10 features
    y = torch.randint(0, 3, (20,))  # 3 classes
    return X, y


def test_predict_architecture(sample_data):
    """Test architecture prediction."""
    X, y = sample_data

    nas = ZeroCostNAS(input_dim=10, output_dim=3, seed=42)

    config = nas.predict_architecture(X, y)

    assert isinstance(config, dict), "Config should be a dictionary"
    assert "hidden_dims" in config, "Config should contain hidden_dims"
    assert "activation_fn_str" in config, "Config should contain activation_fn_str"
    assert isinstance(config["hidden_dims"], list), "hidden_dims should be a list"
    assert isinstance(
        config["activation_fn_str"], str
    ), "activation_fn_str should be a string"


def test_search_with_caching():
    """Test search with caching."""
    # Create temporary directory for cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        X = torch.randn(20, 10)
        y = torch.randint(0, 3, (20,))

        # Initialize NAS with cache
        nas = ZeroCostNAS(input_dim=10, output_dim=3, seed=42, cache_dir=temp_dir)

        # Run search with limited configurations for speed
        result = nas.search(
            X=X,
            y=y,
            depths=[2],
            widths=[32],
            activations=["relu"],
            num_samples=1,
            use_meta_learning=False,
            verbose=False,
        )

        # Check result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "best_config" in result, "Result should contain best_config"
        assert "best_score" in result, "Result should contain best_score"
        assert "search_time" in result, "Result should contain search_time"

        # Check that cache files were created
        cache_files = os.listdir(temp_dir)
        assert len(cache_files) > 0, "Cache files should have been created"


def test_get_best_model(sample_data):
    """Test getting the best model."""
    X, y = sample_data

    nas = ZeroCostNAS(input_dim=10, output_dim=3, seed=42)

    # Run a minimal search
    nas.search(
        X=X,
        y=y,
        depths=[2],
        widths=[32],
        activations=["relu"],
        num_samples=1,
        use_meta_learning=False,
        verbose=False,
    )

    # Get the best model
    model = nas.get_best_model()

    # Test the model
    assert model is not None, "Model should not be None"

    # Test forward pass
    with torch.no_grad():
        output = model(X)

    assert output.shape == (20, 3), "Output shape should match expected dimensions"
