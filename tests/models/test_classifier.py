"""Tests of the FFNN classifier: hidden layers, probabilities and class weights.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote the tests (project 3).
"""

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.utils.class_weight import compute_class_weight

from fys_stk4155_p3.metrics import recall_score
from fys_stk4155_p3.models.classifier import NeuralNetworkClassifier, balanced_class_weights
from fys_stk4155_p3.models.logistic import LogisticRegression
from fys_stk4155_p3.nn.layers import Dense
from fys_stk4155_p3.optimization.optimizers import Adam, Plain

MakeData = Callable[[], tuple[np.ndarray, np.ndarray]]


def make_xor(n_per_cluster: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Four tight clusters at (+-1, +-1), labelled by XOR: not linearly separable."""
    rng = np.random.default_rng(0)
    centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    X = np.vstack([c + 0.1 * rng.standard_normal((n_per_cluster, 2)) for c in centers])
    y = np.repeat(np.array(["same", "same", "diff", "diff"]), n_per_cluster)
    return X, y


def make_imbalanced(n_major: int = 950, n_minor: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Two overlapping 2D Gaussians with a 19:1 class imbalance, labels 0 and 1."""
    rng = np.random.default_rng(1)
    X = np.vstack(
        [rng.normal(0, 1, (n_major, 2)), rng.normal(1.5, 1, (n_minor, 2))],
    )
    y = np.r_[np.zeros(n_major, dtype=int), np.ones(n_minor, dtype=int)]
    return X, y


def make_three_classes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    centers = 3 * np.array([[1, 0], [-1, 0], [0, 1]])
    counts = [30, 15, 5]
    X = np.vstack([c + rng.standard_normal((n, 2)) for c, n in zip(centers, counts, strict=True)])
    y = np.repeat(np.array([3, 7, 9]), counts)
    return X, y


def test_hidden_layer_learns_xor_where_logistic_regression_cannot() -> None:
    X, y = make_xor()
    mlp = NeuralNetworkClassifier(Adam(learning_rate=0.05), hidden_layers=(8,))
    mlp.fit(X, y, epochs=300)
    logistic = LogisticRegression(Adam(learning_rate=0.05))
    logistic.fit(X, y, epochs=300)

    np.testing.assert_array_equal(mlp.predict(X), y)
    assert np.mean(logistic.predict(X) == y) < 0.8


@pytest.mark.parametrize("make_data", [make_xor, make_three_classes])
def test_probabilities_match_predictions(make_data: MakeData) -> None:
    X, y = make_data()
    model = NeuralNetworkClassifier(Adam(learning_rate=0.01), hidden_layers=(6, 4))
    model.fit(X, y, epochs=20, batch_size=8)
    proba = model.predict_proba(X)

    assert model.classes_ is not None
    np.testing.assert_array_equal(model.classes_, np.unique(y))
    assert proba.shape == (len(y), len(model.classes_))
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(X), model.classes_[np.argmax(proba, axis=1)])


def test_n_parameters() -> None:
    X, y = make_three_classes()
    model = NeuralNetworkClassifier(Plain(learning_rate=0.1), hidden_layers=(5, 4))
    model.fit(X, y, epochs=1)
    assert model.n_parameters == (2 * 5 + 5) + (5 * 4 + 4) + (4 * 3 + 3)

    logistic = LogisticRegression(Plain(learning_rate=0.1))
    logistic.fit(X, y, epochs=1)
    assert logistic.n_parameters == 2 * 3 + 3


@pytest.mark.parametrize("make_data", [make_imbalanced, make_three_classes])
def test_balanced_weights_match_scikit_learn(make_data: MakeData) -> None:
    _, y = make_data()
    classes, weights = balanced_class_weights(y)
    np.testing.assert_array_equal(classes, np.unique(y))
    np.testing.assert_allclose(weights, compute_class_weight("balanced", classes=classes, y=y))
    # Weights average to one over the samples, keeping the loss on the same scale
    np.testing.assert_allclose(np.mean(weights[np.searchsorted(classes, y)]), 1.0)


def test_class_weight_equals_explicit_sample_weight() -> None:
    """class_weight="balanced" trains exactly as passing the same weights per sample."""
    X, y = make_three_classes()
    _, weights = balanced_class_weights(y)
    sample_weight = weights[np.searchsorted(np.unique(y), y)]

    by_class = NeuralNetworkClassifier(Adam(0.01), (4,), class_weight="balanced")
    by_class.fit(X, y, epochs=5, batch_size=10)
    by_sample = NeuralNetworkClassifier(Adam(0.01), (4,))
    by_sample.fit(X, y, epochs=5, batch_size=10, sample_weight=sample_weight)

    assert by_class.network is not None and by_sample.network is not None
    for a, b in zip(by_class.network.layers, by_sample.network.layers, strict=True):
        if isinstance(a, Dense) and isinstance(b, Dense):
            np.testing.assert_allclose(a.weights, b.weights)


def test_mapping_class_weight_defaults_to_one() -> None:
    X, y = make_three_classes()
    model = NeuralNetworkClassifier(Plain(0.1), class_weight={9: 5.0})
    model.fit(X, y, epochs=1)
    assert model.class_weights_ is not None
    np.testing.assert_array_equal(model.class_weights_, [1.0, 1.0, 5.0])


def test_balanced_weights_raise_minority_recall() -> None:
    X, y = make_imbalanced()
    recalls = {}
    for class_weight in (None, "balanced"):
        model = LogisticRegression(Adam(learning_rate=0.05), class_weight=class_weight)
        model.fit(X, y, epochs=200, batch_size=100)
        recalls[class_weight] = recall_score(y, model.predict(X))[1]
    assert recalls["balanced"] > recalls[None] + 0.3


def test_class_weights_apply_to_validation_cost() -> None:
    X, y = make_three_classes()
    model = NeuralNetworkClassifier(Adam(0.01), (4,), class_weight="balanced")
    model.fit(X, y, epochs=3, validation_data=(X, y))
    assert model.network is not None and model.class_weights_ is not None

    targets = np.eye(3)[np.searchsorted(np.unique(y), y)]
    sample_weight = model.class_weights_[np.searchsorted(np.unique(y), y)]
    np.testing.assert_allclose(
        model.validation_history[-1], model.network.cost(X, targets, sample_weight)
    )


def test_invalid_use_raises() -> None:
    X, y = make_three_classes()
    with pytest.raises(ValueError, match="hidden_layers"):
        NeuralNetworkClassifier(Plain(0.1), hidden_layers=(4, 0))
    with pytest.raises(ValueError, match="class_weight"):
        NeuralNetworkClassifier(Plain(0.1), class_weight="even").fit(X, y, epochs=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not in the training targets"):
        NeuralNetworkClassifier(Plain(0.1), class_weight={42: 2.0}).fit(X, y, epochs=1)
    with pytest.raises(RuntimeError, match="fit must be called"):
        _ = NeuralNetworkClassifier(Plain(0.1)).n_parameters
