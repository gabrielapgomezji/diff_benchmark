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
        super().__init__()

        if prediction_task == "classification":
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
        return self.head(x)
