"""FFNN class: forward, backprop, fit, predict"""

import copy
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p3.nn.layers import Dense, Layer
from fys_stk4155_p3.nn.losses import (
    Loss,
    Regularization,
    regularization_gradient,
    regularization_penalty,
)
from fys_stk4155_p3.optimization.optimizers import Optimizer
from fys_stk4155_p3.optimization.schedules import (
    constant_schedule,
    exponential_decay,
    time_based_decay,
)

Schedule = Literal["constant", "time_based", "exponential"]

# Maps (initial learning rate, decay strength) to a schedule t -> learning rate
_SCHEDULES: dict[Schedule, Callable[[float, float], Callable[[int], float]]] = {
    "constant": lambda learning_rate, _: constant_schedule(learning_rate),
    "time_based": time_based_decay,
    "exponential": exponential_decay,
}


# (X_val, y_val) or (X_val, y_val, sample_weight_val)
ValidationData = (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]
)


def _as_2d(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert to a float array, turning shape (n_samples,) into (n_samples, 1)."""
    array = np.asarray(array, dtype=np.float64)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _as_weights(
    sample_weight: NDArray[np.float64] | None, n_samples: int, name: str
) -> NDArray[np.float64] | None:
    """Convert per-sample weights to a float array and check that there is one per sample."""
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},), got {weights.shape}.")
    return weights


class FFNN:
    """Feedforward neural network built from a sequence of layers.

    Dense and activation layers are given as separate entries, e.g.
    ``[Dense(1, 50, rng=rng), Sigmoid(), Dense(50, 1, rng=rng), Linear()]``, so the
    forward and backward passes are plain loops over the layers.

    The cost minimized is the mean loss plus a penalty on the weights of all Dense
    layers, C = loss + lambda_ * sum(|W|) for l1 or lambda_ * sum(W**2) for l2. Biases
    are not regularized.

    Args:
        layers: Layers of the network, in the order they are applied.
        loss: Loss function to use for training.
        optimizer: Optimizer to use for training. It serves as a template: each weight
            matrix and bias vector gets its own copy, since the optimizers keep state
            shaped like the parameter they update. Pass a fresh instance, as any state it
            already holds is copied too.
        regularization: Regularization type, either none, l1 or l2.
        lambda_: Regularization strength, must be non-negative.
        seed: Seed for shuffling the training data into minibatches.
        learning_rate_schedule: How the learning rate evolves during ``fit``, starting
            from the optimizer's learning rate: constant, time_based
            (``learning_rate / (1 + lr_decay * t)``) or exponential
            (``learning_rate * exp(-lr_decay * t)``), where ``t`` is the number of
            minibatch updates taken so far in the current call to ``fit``.
        lr_decay: Decay strength of the schedule, must be non-negative.

    Raises:
        ValueError: If ``learning_rate_schedule`` is unknown or ``lr_decay`` is negative.
    """

    def __init__(
        self,
        layers: list[Layer],
        loss: Loss,
        optimizer: Optimizer,
        regularization: Regularization = "none",
        lambda_: float = 0.0,
        seed: int = 42,
        *,
        learning_rate_schedule: Schedule = "constant",
        lr_decay: float = 0.0,
    ) -> None:
        if learning_rate_schedule not in _SCHEDULES:
            raise ValueError(
                f"learning_rate_schedule must be one of {list(_SCHEDULES)}, "
                f"got {learning_rate_schedule!r}."
            )
        if lr_decay < 0:
            raise ValueError(f"lr_decay must be non-negative, got {lr_decay}.")

        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.regularization: Regularization = regularization
        self.lambda_ = lambda_
        self.seed = seed
        self.learning_rate_schedule: Schedule = learning_rate_schedule
        self.lr_decay = lr_decay

        self._dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
        self._optimizers: list[tuple[Optimizer, Optimizer]] = []
        self._ravel_output = False
        self.validation_history: list[float] = []
        self.best_epoch: int | None = None

    def _forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute a forward pass through all layers.

        Each layer caches what it needs for ``_backward``.

        Args:
            inputs: Input data, shape (n_samples, n_features).

        Returns:
            The network output, shape (n_samples, n_outputs).
        """
        a = np.asarray(inputs, dtype=np.float64)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _backward(
        self,
        y_true: NDArray[np.float64],
        output: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> None:
        """Backpropagate the loss gradient through all layers.

        Must follow a call to ``_forward``. Afterwards, each Dense layer holds dC/dW and
        dC/db in ``dweights`` and ``dbiases``. The regularization gradient is not
        included; it is added in ``_update``.

        Args:
            y_true: True targets, same shape as ``output``.
            output: Network output from the preceding ``_forward`` call.
            sample_weight: Optional weight per sample, shape (n_samples,).
        """
        dvalues = self.loss.derivative(y_true, output, sample_weight)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def _update(self, learning_rate: float) -> None:
        """Take one optimizer step on the weights and biases of every Dense layer.

        Must follow a call to ``_backward``.

        Args:
            learning_rate: Learning rate of this step, given by the schedule.
        """
        for layer, (weight_optimizer, bias_optimizer) in zip(
            self._dense_layers, self._optimizers, strict=True
        ):
            weight_optimizer.learning_rate = learning_rate
            bias_optimizer.learning_rate = learning_rate
            dweights = layer.dweights + regularization_gradient(
                layer.weights, self.regularization, self.lambda_
            )
            layer.weights = weight_optimizer.step(layer.weights, dweights)
            layer.biases = bias_optimizer.step(layer.biases, layer.dbiases)

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters, the weights and biases of all Dense layers."""
        return sum(layer.weights.size + layer.biases.size for layer in self._dense_layers)

    def _penalty(self) -> np.float64:
        """Regularization penalty on the current weights."""
        return regularization_penalty(
            [layer.weights for layer in self._dense_layers], self.regularization, self.lambda_
        )

    def cost(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> np.float64:
        """Compute the cost, the (weighted) mean loss plus the regularization penalty.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).
            sample_weight: Optional non-negative weight per sample, shape (n_samples,).

        Returns:
            The cost on the given data.
        """
        output = self._forward(_as_2d(X))
        return np.float64(self.loss(_as_2d(y), output, sample_weight) + self._penalty())

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int = 100,
        batch_size: int | None = None,
        validation_data: ValidationData | None = None,
        patience: int | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> list[float]:
        """Train the network with (stochastic) gradient descent.

        Each epoch shuffles the training data and takes one optimizer step per
        minibatch. With ``batch_size=None`` the whole dataset is one batch, giving plain
        (full-batch) gradient descent. Calling ``fit`` again continues training from the
        current weights, with fresh optimizer state and a restarted learning rate
        schedule.

        The training cost recorded for each epoch is the running mean of the minibatch
        costs (loss plus penalty) taken during that epoch, the same as the training loss
        Keras reports. It costs no extra forward pass, which matters for large datasets,
        but lags behind the cost at the end of the epoch since the weights change during
        it. With full-batch training it is the cost before the epoch's single step.

        If ``validation_data`` is given, the cost on it is recorded after each epoch in
        ``validation_history``, which is reset at the start of every call. The validation
        data is never used to update the weights. A third element in ``validation_data``
        weights the validation samples, e.g. with the same class weights as the training
        data so that early stopping tracks the cost being minimized.

        With ``patience`` set, training stops early once the validation cost has not
        improved for ``patience`` epochs in a row, and the weights from the epoch with
        the lowest validation cost are restored, also when no early stop happens. That
        epoch (counting from 1) is stored in ``best_epoch``. The histories still cover
        every epoch trained, so they may be shorter than ``epochs``.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).
            y: Targets, shape (n_samples,) or (n_samples, n_outputs).
            epochs: Maximum number of passes over the training data.
            batch_size: Number of samples per minibatch, or None for full-batch.
            validation_data: Optional (X_val, y_val) shaped like ``X`` and ``y``, or
                (X_val, y_val, sample_weight_val).
            patience: Number of epochs without improvement in the validation cost before
                stopping early, or None to always train for ``epochs`` epochs. Requires
                ``validation_data``.
            sample_weight: Optional non-negative weight per training sample, shape
                (n_samples,), e.g. class weights against imbalance. See ``losses``.

        Returns:
            The training cost of each epoch, the running mean over its minibatches.

        Raises:
            ValueError: If ``X`` and ``y``, or ``X_val`` and ``y_val``, have different
                numbers of samples, if a sample weight array has the wrong shape, if
                ``epochs``, ``batch_size`` or ``patience`` is not positive, or if
                ``patience`` is given without ``validation_data``.

        LLM-assisted
        ------------
        Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
        Role: Completed the author's draft of the learning rate schedules and early
            stopping (snippet). Added the sample weights and the running-mean training
            cost (project 3).
        """
        X = _as_2d(X)
        self._ravel_output = np.ndim(y) == 1
        y = _as_2d(y)
        n_samples = X.shape[0]

        if y.shape[0] != n_samples:
            raise ValueError(
                f"X and y must have the same number of samples, got {n_samples} and {y.shape[0]}."
            )
        weights = _as_weights(sample_weight, n_samples, "sample_weight")
        if validation_data is not None:
            X_val, y_val = _as_2d(validation_data[0]), _as_2d(validation_data[1])
            if y_val.shape[0] != X_val.shape[0]:
                raise ValueError(
                    "X_val and y_val must have the same number of samples, "
                    f"got {X_val.shape[0]} and {y_val.shape[0]}."
                )
            weights_val = _as_weights(
                validation_data[2] if len(validation_data) == 3 else None,
                X_val.shape[0],
                "sample_weight_val",
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}.")
        if batch_size is None:
            batch_size = n_samples
        elif batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if patience is not None:
            if validation_data is None:
                raise ValueError("patience requires validation_data.")
            if patience <= 0:
                raise ValueError(f"patience must be positive, got {patience}.")

        self._optimizers = [
            (copy.deepcopy(self.optimizer), copy.deepcopy(self.optimizer))
            for _ in self._dense_layers
        ]
        schedule = _SCHEDULES[self.learning_rate_schedule](
            self.optimizer.learning_rate, self.lr_decay
        )
        rng = np.random.default_rng(self.seed)
        history = []
        self.validation_history = []
        self.best_epoch = None
        best_cost, best_epoch = np.inf, 0
        best_parameters: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        step = 0

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(n_samples)
            epoch_cost = 0.0
            for start in range(0, n_samples, batch_size):
                batch = indices[start : start + batch_size]
                batch_weights = None if weights is None else weights[batch]
                output = self._forward(X[batch])
                batch_cost = self.loss(y[batch], output, batch_weights) + self._penalty()
                epoch_cost += float(batch_cost) * len(batch)
                self._backward(y[batch], output, batch_weights)
                self._update(schedule(step))
                step += 1

            history.append(epoch_cost / n_samples)
            if validation_data is None:
                continue
            self.validation_history.append(float(self.cost(X_val, y_val, weights_val)))

            if patience is None:
                continue
            if self.validation_history[-1] < best_cost:
                best_cost = self.validation_history[-1]
                best_epoch = epoch
                best_parameters = [
                    (layer.weights.copy(), layer.biases.copy()) for layer in self._dense_layers
                ]
            elif epoch - best_epoch >= patience:
                break

        if best_parameters:
            self.best_epoch = best_epoch
            for layer, (weights, biases) in zip(self._dense_layers, best_parameters, strict=True):
                layer.weights, layer.biases = weights, biases

        return history

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the network output for the given inputs.

        For classification losses this is the logits, since the sigmoid or softmax is
        part of the loss.

        Args:
            X: Input data, shape (n_samples,) or (n_samples, n_features).

        Returns:
            The network output, shape (n_samples,) if ``fit`` was given 1D targets and
            (n_samples, n_outputs) otherwise.
        """
        output = self._forward(_as_2d(X))
        return output.ravel() if self._ravel_output else output
