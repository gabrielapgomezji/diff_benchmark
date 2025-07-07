from dipy.reconst.mapmri import MapmriModel


def get_model_configs(gtab):
    """
    Generate a dictionary of model configurations for diffusion MRI preprocessing.
    This function creates a set of predefined configurations for different models
    based on the input gradient table (`gtab`). The configurations include default
    models, baseline models, and various constrained models (e.g., Laplacian regularization,
    positivity constraints, and global constraints). Currently, only the default model
    configuration is active, while other configurations are commented out.
    Args:
        gtab (object): The gradient table used for model initialization.
    Returns:
        dict: A dictionary where keys are model names (strings) and values are
              instances of the corresponding model initialized with the given `gtab`.
    """

    configs = {
        # Default + baseline models
        "mapmri_default": MapmriModel(gtab=gtab)
    }

    # # Dummy
    # configs['dummy'] = DummyModel(gtab)

    # # Laplacian-only models
    # for weight in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    #     configs[f'mapmri_laplacian_{weight}'] = MapmriModel(
    #         gtab=gtab,
    #         radial_order=6,
    #         laplacian_regularization=True,
    #         laplacian_weighting=weight
    #     )

    # # Positivity-only
    # configs['mapmri_positivity'] = MapmriModel(
    #     gtab=gtab,
    #     radial_order=6,
    #     positivity_constraint=True
    # )

    # # Laplacian + Positivity
    # for weight in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    #     configs[f'mapmri_lap+pos_{weight}'] = MapmriModel(
    #         gtab=gtab,
    #         radial_order=6,
    #         laplacian_regularization=True,
    #         positivity_constraint=True,
    #         laplacian_weighting=weight
    #     )

    # # Full constraint model
    # configs['mapmri_plus'] = MapmriModel(
    #     gtab=gtab,
    #     radial_order=6,
    #     positivity_constraint=True,
    #     global_constraints=True
    # )

    return configs
