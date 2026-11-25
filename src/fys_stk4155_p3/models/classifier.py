"""Feed-forward neural network classifier with a Scikit-Learn style interface.

Wraps ``FFNN`` with what a classifier needs on top of the network: encoding the class
labels as targets, class weights against imbalance, and turning the logits into class
probabilities. It has the same ``fit`` / ``predict`` / ``predict_proba`` interface as the
Scikit-Learn and XGBoost classifiers, so all methods can be evaluated by the same code.

For two classes the network has a single output and is trained with the binary cross
entropy (sigmoid); for more it has one output per class and is trained with the
categorical cross entropy (softmax). With no hidden layers this is (multinomial)
logistic regression, see ``LogisticRegression``.

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Generalized the author's ``LogisticRegression`` (project 2) to hidden layers and
    added the class weights, following the author's design.
"""

from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fys_stk4155_p3.nn.activations import ReLU, sigmoid, softmax
from fys_stk4155_p3.nn.layers import Dense, Initializer, Layer, Linear
from fys_stk4155_p3.nn.losses import BinaryCrossEntropy, CategoricalCrossEntropy, Regularization
from fys_stk4155_p3.nn.network import FFNN, Schedule, ValidationData
from fys_stk4155_p3.optimization.optimizers import Optimizer

ClassWeight = Literal["balanced"] | Mapping[object, float] | None


def balanced_class_weights(y: NDArray) -> tuple[NDArray, NDArray[np.float64]]:
    """Class weights inversely proportional to the class frequencies.

    The weight of class c is n / (K * n_c), with n samples, K classes and n_c samples in
    class c, the same as ``class_weight="balanced"`` in Scikit-Learn. Every class then
    carries the same total weight, and the weights average to one over the samples.

    Args:
        y: Class labels, shape (n_samples,).

    Returns:
        The sorted classes and the weight of each, both shape (n_classes,).
    """
    classes, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
    return classes, len(y) / (len(classes) * counts)


class NeuralNetworkClassifier:
    """Binary or multiclass classifier built on a feed-forward neural network.

    The network is built in ``fit``, once the number of features and classes is known.
    The classes are the sorted unique labels of the training targets, so the labels may
    be any integers or strings; ``predict`` returns labels from the same set.

    Args:
        optimizer: Optimizer to use for training, passed on to ``FFNN``. Pass a fresh
            instance.
        hidden_layers: Number of nodes in each hidden layer; empty for logistic regression.
        activation: Activation layer of the hidden layers, called once per hidden layer.
        initialization: Weight initialization of the hidden layers. The output layer is
            always initialized with Xavier/Glorot, since it is linear.
        regularization: Regularization type, either none, l1 or l2.
        lambda_: Regularization strength, must be non-negative.
        class_weight: None for equal weights, "balanced" for weights inversely
            proportional to the class frequencies (see ``balanced_class_weights``), or a
            mapping from class label to weight, where missing classes get weight 1.
        seed: Seed for the initial weights and for shuffling into minibatches.
        learning_rate_schedule: Learning rate schedule, see ``FFNN``.
        lr_decay: Decay strength of the schedule, see ``FFNN``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        hidden_layers: tuple[int, ...] = (),
        activation: Callable[[], Layer] = ReLU,
        initialization: Initializer = "he",
        regularization: Regularization = "none",
        lambda_: float = 0.0,
        class_weight: ClassWeight = None,
        seed: int = 42,
        *,
        learning_rate_schedule: Schedule = "constant",
        lr_decay: float = 0.0,
    ) -> None:
        if any(n_nodes <= 0 for n_nodes in hidden_layers):
            raise ValueError(f"hidden_layers must be positive, got {hidden_layers}.")
        self.optimizer = optimizer
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.initialization: Initializer = initialization
        self.regularization: Regularization = regularization
        self.lambda_ = lambda_
        self.class_weight = class_weight
        self.seed = seed
        self.learning_rate_schedule: Schedule = learning_rate_schedule
        self.lr_decay = lr_decay

        self.classes_: NDArray | None = None
        self.class_weights_: NDArray[np.float64] | None = None
        self.network: FFNN | None = None

    @property
    def validation_history(self) -> list[float]:
        """Validation cost after each epoch of the last ``fit``, see ``FFNN.fit``."""
        return [] if self.network is None else self.network.validation_history

    @property
    def best_epoch(self) -> int | None:
        """Epoch whose weights were restored by early stopping, see ``FFNN.fit``."""
        return None if self.network is None else self.network.best_epoch

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters of the network."""
        if self.network is None:
            raise RuntimeError("fit must be called before counting the parameters.")
        return self.network.n_parameters

    def _class_indices(self, y: NDArray) -> NDArray[np.int_]:
        """Index of each label in ``classes_``."""
        assert self.classes_ is not None
        y = np.asarray(y).ravel()
        indices = np.searchsorted(self.classes_, y)
        if np.any(indices >= len(self.classes_)) or np.any(
            self.classes_[np.minimum(indices, len(self.classes_) - 1)] != y
        ):
            raise ValueError("y contains labels that were not in the training targets.")
        return indices

    def _encode(self, indices: NDArray[np.int_]) -> NDArray[np.float64]:
        """Turn class indices into network targets: (n, 1) in {0, 1}, or one-hot (n, K)."""
        assert self.classes_ is not None
        if len(self.classes_) == 2:
            return indices.reshape(-1, 1).astype(np.float64)
        return np.eye(len(self.classes_))[indices]

    def _resolve_class_weights(self, y: NDArray) -> NDArray[np.float64] | None:
        """Weight of each class in ``classes_``, or None for equal weights."""
        assert self.classes_ is not None
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    f"class_weight must be 'balanced', a mapping or None, "
                    f"got {self.class_weight!r}."
                )
            return balanced_class_weights(y)[1]
        unknown = set(self.class_weight) - set(self.classes_.tolist())
        if unknown:
            raise ValueError(f"class_weight has classes not in the training targets: {unknown}.")
        return np.array([float(self.class_weight.get(c, 1.0)) for c in self.classes_.tolist()])

    def _sample_weights(
        self, indices: NDArray[np.int_], sample_weight: NDArray[np.float64] | None
    ) -> NDArray[np.float64] | None:
        """Combine the class weights and any given sample weights by multiplication."""
        weights = None if self.class_weights_ is None else self.class_weights_[indices]
        if sample_weight is None:
            return weights
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        return sample_weight if weights is None else weights * sample_weight

    def _build_network(self, n_features: int, n_outputs: int, binary: bool) -> FFNN:
        rng = np.random.default_rng(self.seed)
        layers: list[Layer] = []
        n_inputs = n_features
        for n_nodes in self.hidden_layers:
            layers += [Dense(n_inputs, n_nodes, self.initialization, rng=rng), self.activation()]
            n_inputs = n_nodes
        layers += [Dense(n_inputs, n_outputs, "xavier", rng=rng), Linear()]
        return FFNN(
            layers,
            BinaryCrossEntropy() if binary else CategoricalCrossEntropy(),
            self.optimizer,
            self.regularization,
            self.lambda_,
            self.seed,
            learning_rate_schedule=self.learning_rate_schedule,
            lr_decay=self.lr_decay,
        )

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray,
        epochs: int = 100,
        batch_size: int | None = None,
        validation_data: tuple[NDArray[np.float64], NDArray] | None = None,
        patience: int | None = None,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> list[float]:
        """Train the model from scratch on the given data.

        The class weights are computed from the training labels and applied to the
        validation samples too, so the validation cost used for early stopping is the
        same weighted cost that training minimizes.

        Args:
            X: Input data, shape (n_samples, n_features).
            y: Class labels, shape (n_samples,), with at least two distinct classes.
            epochs: Maximum number of passes over the training data.
            batch_size: Number of samples per minibatch, or None for full-batch.
            validation_data: Optional pair (X_val, y_val) of inputs and class labels.
            patience: Early stopping patience in epochs, see ``FFNN.fit``.
            sample_weight: Optional non-negative weight per training sample, shape
                (n_samples,), multiplied with the class weights.

        Returns:
            The training cost of each epoch, see ``FFNN.fit``.

        Raises:
            ValueError: If ``y`` has fewer than two classes, ``y_val`` has labels not in
                ``y``, or ``class_weight`` is invalid.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError(f"y must contain at least two classes, got {len(self.classes_)}.")
        self.class_weights_ = self._resolve_class_weights(y)

        binary = len(self.classes_) == 2
        self.network = self._build_network(X.shape[1], 1 if binary else len(self.classes_), binary)

        indices = self._class_indices(y)
        encoded_validation: ValidationData | None = None
        if validation_data is not None:
            X_val, y_val = validation_data
            indices_val = self._class_indices(y_val)
            encoded_validation = (
                X_val,
                self._encode(indices_val),
                self._sample_weights(indices_val, None),
            )
        return self.network.fit(
            X,
            self._encode(indices),
            epochs,
            batch_size,
            validation_data=encoded_validation,
            patience=patience,
            sample_weight=self._sample_weights(indices, sample_weight),
        )

    def decision_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Network output (logits), shape (n_samples, 1) for two classes, else (n, K)."""
        if self.network is None:
            raise RuntimeError("fit must be called before predicting.")
        return self.network.predict(X)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predicted probability of each class, in the order of ``classes_``.

        For two classes this has two columns, [1 - p, p], as in Scikit-Learn, so that
        evaluation code needs no special case.

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            The class probabilities, shape (n_samples, n_classes); each row sums to 1.
        """
        logits = self.decision_function(X)
        if logits.shape[1] == 1:
            p = sigmoid(logits)
            return np.hstack([1 - p, p])
        return softmax(logits)

    def predict(self, X: NDArray[np.float64]) -> NDArray:
        """Predicted class label of each sample, the class with the highest probability.

        Args:
            X: Input data, shape (n_samples, n_features).

        Returns:
            The predicted labels, shape (n_samples,).
        """
        logits = self.decision_function(X)
        assert self.classes_ is not None
        if logits.shape[1] == 1:
            return self.classes_[(logits[:, 0] > 0).astype(np.int_)]
        return self.classes_[np.argmax(logits, axis=1)]
