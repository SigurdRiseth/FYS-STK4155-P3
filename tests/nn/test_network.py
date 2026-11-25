"""Backpropagation and training in FFNN compared with automatic differentiation in JAX.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote the module, including the JAX reference costs and the tests. Added the sample
    weights, the minibatch trajectory, the duplication and the training cost tests
    (project 3).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fys_stk4155_p3.nn.activations import Activation, LeakyReLU, ReLU, Sigmoid
from fys_stk4155_p3.nn.layers import Dense, Layer, Linear
from fys_stk4155_p3.nn.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Loss,
    MeanSquaredError,
    Regularization,
)
from fys_stk4155_p3.nn.network import FFNN
from fys_stk4155_p3.optimization.optimizers import Plain

jax.config.update("jax_enable_x64", True)

N_SAMPLES, N_FEATURES, HIDDEN_LAYERS = 8, 3, [5, 4]

JAX_ACTIVATIONS: dict[type[Activation], Callable[[jax.Array], jax.Array]] = {
    Sigmoid: jax.nn.sigmoid,
    ReLU: jax.nn.relu,
    LeakyReLU: lambda z: jnp.where(z > 0, z, 0.01 * z),
}

# The JAX losses take weights of shape (n_samples, 1); the mean is over n, not sum(w).
JaxLoss = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


def mse(y: jax.Array, output: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.mean(w * (output - y) ** 2)


def binary_cross_entropy(y: jax.Array, logits: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.mean(w * (jnp.logaddexp(0, logits) - y * logits))


def categorical_cross_entropy(y: jax.Array, logits: jax.Array, w: jax.Array) -> jax.Array:
    return -jnp.mean(w[:, 0] * jnp.sum(y * jax.nn.log_softmax(logits), axis=1))


# Our loss, its JAX counterpart and the number of outputs
LOSSES: dict[str, tuple[type[Loss], JaxLoss, int]] = {
    "mse": (MeanSquaredError, mse, 2),
    "bce": (BinaryCrossEntropy, binary_cross_entropy, 1),
    "cce": (CategoricalCrossEntropy, categorical_cross_entropy, 3),
}

Parameters = list[tuple[np.ndarray, np.ndarray]]


def make_targets(loss: str, n_outputs: int, rng: np.random.Generator) -> np.ndarray:
    if loss == "mse":
        return rng.standard_normal((N_SAMPLES, n_outputs))
    if loss == "bce":
        return rng.integers(0, 2, (N_SAMPLES, 1)).astype(float)
    return np.eye(n_outputs)[rng.integers(0, n_outputs, N_SAMPLES)]


def make_layers(
    activation: type[Activation], n_outputs: int, rng: np.random.Generator
) -> tuple[list[Layer], list[Dense]]:
    """Hidden layers of HIDDEN_LAYERS nodes and a linear output, with nonzero biases."""
    layers: list[Layer] = []
    n_inputs = N_FEATURES
    for n_neurons in HIDDEN_LAYERS:
        layers += [Dense(n_inputs, n_neurons, rng=rng), activation()]
        n_inputs = n_neurons
    layers += [Dense(n_inputs, n_outputs, rng=rng), Linear()]
    dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
    # Nonzero biases, so that the bias gradients are tested away from the initialization
    for layer in dense_layers:
        layer.biases = rng.standard_normal(layer.biases.shape)
    return layers, dense_layers


def make_jax_cost(
    activation: type[Activation],
    jax_loss: JaxLoss,
    regularization: Regularization,
    lambda_: float,
) -> Callable[[Parameters, np.ndarray, np.ndarray, np.ndarray], jax.Array]:
    """The FFNN cost written in JAX, as a function of the parameters and the data."""

    def cost(parameters: Parameters, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> jax.Array:
        a = jnp.asarray(X)
        for i, (weights, biases) in enumerate(parameters):
            z = a @ weights + biases
            a = z if i == len(parameters) - 1 else JAX_ACTIVATIONS[activation](z)
        penalty = {
            "none": 0.0,
            "l1": lambda_ * sum(jnp.sum(jnp.abs(weights)) for weights, _ in parameters),
            "l2": lambda_ * sum(jnp.sum(weights**2) for weights, _ in parameters),
        }[regularization]
        return jax_loss(jnp.asarray(y), a, jnp.asarray(w).reshape(-1, 1)) + penalty

    return cost


def assert_parameters_close(dense_layers: list[Dense], expected: Parameters) -> None:
    for layer, (weights, biases) in zip(dense_layers, expected, strict=True):
        np.testing.assert_allclose(layer.weights, weights, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(layer.biases, biases, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("regularization", ["none", "l1", "l2"])
@pytest.mark.parametrize("loss", list(LOSSES))
@pytest.mark.parametrize("activation", list(JAX_ACTIVATIONS))
def test_gradient_matches_jax(
    activation: type[Activation], loss: str, regularization: Regularization, weighted: bool
) -> None:
    """One full-batch plain GD step with learning rate 1 gives W - dC/dW, compared with jax.grad."""
    loss_class, jax_loss, n_outputs = LOSSES[loss]
    lambda_ = 0.0 if regularization == "none" else 0.1
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets(loss, n_outputs, rng)
    sample_weight = rng.uniform(0, 3, N_SAMPLES) if weighted else None

    layers, dense_layers = make_layers(activation, n_outputs, rng)
    parameters = [(layer.weights.copy(), layer.biases.copy()) for layer in dense_layers]
    cost = make_jax_cost(activation, jax_loss, regularization, lambda_)
    w = np.ones(N_SAMPLES) if sample_weight is None else sample_weight
    gradient = jax.grad(cost)(parameters, X, y, w)

    network = FFNN(layers, loss_class(), Plain(learning_rate=1.0), regularization, lambda_)
    network.fit(X, y, epochs=1, sample_weight=sample_weight)

    expected = [(W - dW, b - db) for (W, b), (dW, db) in zip(parameters, gradient, strict=True)]
    assert_parameters_close(dense_layers, expected)


@pytest.mark.parametrize("loss", list(LOSSES))
def test_minibatch_training_matches_jax(loss: str) -> None:
    """Several epochs of weighted, regularized minibatch GD follow the same trajectory as JAX.

    The JAX loop draws the minibatches with the same seeded permutations as ``FFNN.fit``,
    so this checks the shuffling, the batch slicing of the targets and sample weights, and
    the parameter updates over many steps, not only a single gradient.
    """
    loss_class, jax_loss, n_outputs = LOSSES[loss]
    learning_rate, epochs, batch_size, lambda_, seed = 0.1, 4, 3, 0.01, 7
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets(loss, n_outputs, rng)
    sample_weight = rng.uniform(0, 3, N_SAMPLES)

    layers, dense_layers = make_layers(Sigmoid, n_outputs, rng)
    parameters = [(layer.weights.copy(), layer.biases.copy()) for layer in dense_layers]
    gradient = jax.grad(make_jax_cost(Sigmoid, jax_loss, "l2", lambda_))
    shuffle = np.random.default_rng(seed)
    for _ in range(epochs):
        indices = shuffle.permutation(N_SAMPLES)
        for start in range(0, N_SAMPLES, batch_size):
            batch = indices[start : start + batch_size]
            grads = gradient(parameters, X[batch], y[batch], sample_weight[batch])
            parameters = [
                (W - learning_rate * np.asarray(dW), b - learning_rate * np.asarray(db))
                for (W, b), (dW, db) in zip(parameters, grads, strict=True)
            ]

    network = FFNN(layers, loss_class(), Plain(learning_rate), "l2", lambda_, seed=seed)
    network.fit(X, y, epochs=epochs, batch_size=batch_size, sample_weight=sample_weight)

    assert_parameters_close(dense_layers, parameters)


@pytest.mark.parametrize("loss", list(LOSSES))
def test_weight_two_equals_duplicated_sample(loss: str) -> None:
    """Weight 2 on one sample and 0 on another equals duplicating the first, dropping the other.

    The weights sum to n, so both datasets have the same normalization, and the cost and
    a full-batch gradient step must agree exactly. Needs no reference implementation.
    """
    loss_class, _, n_outputs = LOSSES[loss]
    rng = np.random.default_rng(2)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets(loss, n_outputs, rng)
    sample_weight = np.ones(N_SAMPLES)
    sample_weight[0], sample_weight[1] = 2.0, 0.0
    duplicated = np.r_[0, 0, 2:N_SAMPLES]

    networks, dense_layers = [], []
    for _ in range(2):
        layers, dense = make_layers(ReLU, n_outputs, np.random.default_rng(3))
        networks.append(FFNN(layers, loss_class(), Plain(learning_rate=0.5), "l2", 0.1))
        dense_layers.append(dense)
    weighted, copied = networks

    np.testing.assert_allclose(
        weighted.cost(X, y, sample_weight), copied.cost(X[duplicated], y[duplicated])
    )
    weighted.fit(X, y, epochs=1, sample_weight=sample_weight)
    copied.fit(X[duplicated], y[duplicated], epochs=1)
    assert_parameters_close(
        dense_layers[0], [(layer.weights, layer.biases) for layer in dense_layers[1]]
    )


def test_unit_weights_equal_no_weights() -> None:
    rng = np.random.default_rng(4)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets("cce", 3, rng)
    layers, _ = make_layers(ReLU, 3, rng)
    network = FFNN(layers, CategoricalCrossEntropy(), Plain(learning_rate=0.1))
    assert network.cost(X, y, np.ones(N_SAMPLES)) == network.cost(X, y)


def test_full_batch_history_is_cost_before_step() -> None:
    """With full-batch GD the running-mean training cost is the cost before each step."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets("bce", 1, rng)
    sample_weight = rng.uniform(0, 3, N_SAMPLES)
    layers, _ = make_layers(ReLU, 1, rng)
    network = FFNN(layers, BinaryCrossEntropy(), Plain(learning_rate=0.1), "l2", 0.1)

    costs = []
    for _ in range(3):
        costs.append(float(network.cost(X, y, sample_weight)))
        history = network.fit(X, y, epochs=1, sample_weight=sample_weight)
        np.testing.assert_allclose(history, costs[-1:])


def test_weighted_validation_cost() -> None:
    """A third element of validation_data weights the validation cost."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets("bce", 1, rng)
    weights_val = rng.uniform(0, 3, N_SAMPLES)
    layers, _ = make_layers(ReLU, 1, rng)
    network = FFNN(layers, BinaryCrossEntropy(), Plain(learning_rate=0.1))
    network.fit(X, y, epochs=2, validation_data=(X, y, weights_val))
    np.testing.assert_allclose(network.validation_history[-1], network.cost(X, y, weights_val))


def test_n_parameters() -> None:
    layers, _ = make_layers(ReLU, 2, np.random.default_rng(0))
    network = FFNN(layers, MeanSquaredError(), Plain(learning_rate=0.1))
    assert network.n_parameters == (3 * 5 + 5) + (5 * 4 + 4) + (4 * 2 + 2)


@pytest.mark.parametrize(
    "sample_weight", [np.ones(N_SAMPLES - 1), -np.ones(N_SAMPLES), np.full(N_SAMPLES, np.nan)]
)
def test_invalid_sample_weight_raises(sample_weight: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = make_targets("mse", 2, rng)
    layers, _ = make_layers(ReLU, 2, rng)
    network = FFNN(layers, MeanSquaredError(), Plain(learning_rate=0.1))
    with pytest.raises(ValueError, match="sample_weight"):
        network.fit(X, y, epochs=1, sample_weight=sample_weight)
