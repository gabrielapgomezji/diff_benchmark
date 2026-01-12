from torch import nn


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation=nn.ReLU,
    ):
        """
        Initialize a prediction head with optional hidden layers.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output predictions.
            hidden_dims (list[int] | None, optional): List of dimensions for hidden layers.
                If None or empty, creates a linear layer directly from input to output.
                Defaults to None.
            dropout (float, optional): Dropout rate to apply after each hidden layer.
                If 0, no dropout is applied. Defaults to 0.0.
            activation (type, optional): Activation function class to use between hidden layers.
                Defaults to nn.ReLU.
        Returns:
            None
        """
        super().__init__()

        hidden_dims = hidden_dims or []

        layers = []
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
        """
        Forward pass of the prediction head.
        Args:
            x: Input tensor to be passed through the prediction head network.
        Returns:
            Output tensor from the prediction head network.
        """

        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        prediction_task: str,
        num_classes: int | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        """
        Initialize a prediction head for classification or regression tasks.
        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            prediction_task (str): The type of prediction task, either "classification" or "regression".
            num_classes (int | None, optional): The number of output classes for classification.
                Required if prediction_task is "classification". Defaults to None.
            hidden_dims (list[int] | None, optional): The dimensions of hidden layers in the MLP.
                Defaults to None.
            dropout (float, optional): The dropout rate to apply in the MLP layers.
                Defaults to 0.0.
        Raises:
            ValueError: If prediction_task is "classification" and num_classes is None.
        Note:
            If prediction_task is neither "classification" nor "regression", a warning message
            is printed but no exception is raised.
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
            print("No prediction task specified yet")
            # raise ValueError(f"Unknown task: {prediction_task}")

    def forward(self, x):
        """
        Forward pass of the prediction head.
        Args:
            x: Input tensor to the prediction head.
        Returns:
            Output tensor from the prediction head.
        """

        return self.head(x)


def build_prediction_head(
    embedding_dim: int,
    prediction_task: str,
    # num_classes: int | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Build a prediction head module for a given embedding dimension and task.
    Args:
        embedding_dim (int): The dimension of the input embeddings.
        prediction_task (str): The type of prediction task (e.g., 'classification', 'regression').
        num_classes (int | None, optional): The number of output classes. Required for classification tasks.
            Defaults to None.
        hidden_dims (list[int] | None, optional): List of hidden layer dimensions for the prediction head.
            If None, uses default architecture. Defaults to None.
        dropout (float, optional): Dropout rate to apply between layers. Defaults to 0.0.
    Returns:
        nn.Module: A prediction head module configured for the specified task and parameters.
    """
    num_classes = 2 if prediction_task == "binary_classification" else 1 if prediction_task == "regression" else Exception(f"Unknown task: {prediction_task}. Prediction Head only implemented for binary classification and regression tasks.")
    
    return PredictionHead(
        embedding_dim=embedding_dim,
        prediction_task=prediction_task,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
