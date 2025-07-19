import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from src.train import load_config, train_model

CONFIG_PATH = "config/config.json"

def test_config_file_exists():
    assert os.path.exists(CONFIG_PATH), "Config file missing"

def test_config_values():
    config = load_config(CONFIG_PATH)
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

def test_model_accuracy():
    config = load_config(CONFIG_PATH)
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model(X, y, config)
    score = model.score(X, y)
    assert score > 0.8, f"Accuracy too low: {score}"
