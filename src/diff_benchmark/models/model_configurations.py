from diff_benchmark.models.cca import CanonicalCorrelationRegressor


def get_model(name: str, config: dict):
    name = name.lower()

    if name == "cca":
        return CanonicalCorrelationRegressor(n_components=config["n_components"])

    # elif name == "other_model":
    #     return OtherModelClass(param1=config["param1"], ...)

    else:
        raise ValueError(f"Unknown model name: {name}")
