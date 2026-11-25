"""Logistic regression as an FFNN with no hidden layers.

The baseline classifier: a single Dense layer with a linear output, trained with the
binary cross entropy for two classes (sigmoid) and the categorical cross entropy for more
(softmax, multinomial logistic regression). Since the sigmoid and softmax are part of the
losses, this is the same model as Sections 5.3 and 5.6 of Hjorth-Jensen (2026).

LLM-assisted
------------
Tool: Claude (claude-opus-5-5, Claude Code CLI, November 2026)
Role: Wrote the module from the author's specification; tested by the author. Reduced
    to a special case of ``NeuralNetworkClassifier`` (project 3).
"""

from fys_stk4155_p3.models.classifier import ClassWeight, NeuralNetworkClassifier
from fys_stk4155_p3.nn.losses import Regularization
from fys_stk4155_p3.optimization.optimizers import Optimizer


class LogisticRegression(NeuralNetworkClassifier):
    """Binary or multinomial logistic regression, trained with gradient descent.

    A ``NeuralNetworkClassifier`` without hidden layers; see it for ``fit``,
    ``predict_proba`` and ``predict``.

    Args:
        optimizer: Optimizer to use for training, passed on to ``FFNN``. Pass a fresh
            instance.
        regularization: Regularization type, either none, l1 or l2.
        lambda_: Regularization strength, must be non-negative.
        class_weight: None, "balanced" or a mapping from class to weight, see
            ``NeuralNetworkClassifier``.
        seed: Seed for the initial weights and for shuffling into minibatches.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        regularization: Regularization = "none",
        lambda_: float = 0.0,
        class_weight: ClassWeight = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            optimizer,
            hidden_layers=(),
            regularization=regularization,
            lambda_=lambda_,
            class_weight=class_weight,
            seed=seed,
        )
