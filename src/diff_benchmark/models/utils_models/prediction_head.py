from torch import nn

from diff_benchmark.utils.logger import setup_logger

logger = setup_logger(__name__)


class MLPHead(nn.Module):
    """MLP prediction head with optional hidden layers, LayerNorm input, and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation=nn.ReLU,
    ):
        super().__init__()

        hidden_dims = hidden_dims or []

        layers = []
        # Normalize input features before the first linear projection
        layers.append(nn.LayerNorm(input_dim))
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    """Routes embedding to an ``MLPHead`` sized for the prediction task."""

    def __init__(
        self,
        embedding_dim: int,
        prediction_task: str,
        num_classes: int | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            embedding_dim (int): Dimensionality of the input embeddings.
            prediction_task (str): ``"binary_classification"`` or ``"regression"``.
            num_classes (int | None): Number of output classes; required for classification.
            hidden_dims (list[int] | None): Hidden-layer dimensions for the MLP.
            dropout (float): Dropout rate applied between MLP layers.

        Raises:
            ValueError: If ``prediction_task == "binary_classification"`` and
                ``num_classes`` is ``None``.
        """
        super().__init__()

        if prediction_task == "binary_classification":
            if num_classes is None:
                raise ValueError("num_classes required for classification")

            self.head = MLPHead(
                input_dim=embedding_dim,
                output_dim=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        elif prediction_task == "regression":
            self.head = MLPHead(
                input_dim=embedding_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )

        else:
            logger.warning(
                f"Unknown task: {prediction_task}. Prediction Head only implemented for binary classification and regression tasks."
            )

    def forward(self, x):
        return self.head(x)


def build_prediction_head(
    embedding_dim: int,
    prediction_task: str,
    # num_classes: int | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a ``PredictionHead`` for the given embedding dimension and task.

    Args:
        embedding_dim (int): Dimension of the input embeddings.
        prediction_task (str): ``"binary_classification"`` or ``"regression"``.
        hidden_dims (list[int] | None): Hidden-layer dimensions for the MLP.
        dropout (float): Dropout rate applied between MLP layers.

    Returns:
        nn.Module: Configured ``PredictionHead``.
    """
    num_classes = (
        2
        if prediction_task == "binary_classification"
        else (
            1
            if prediction_task == "regression"
            else Exception(
                f"Unknown task: {prediction_task}. Prediction Head only implemented for binary classification and regression tasks."
            )
        )
    )

    return PredictionHead(
        embedding_dim=embedding_dim,
        prediction_task=prediction_task,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
