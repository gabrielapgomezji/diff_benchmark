import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_history(path="history.json"):
    """
    Load training history from a JSON file.
    Args:
        path (str): The file path to the JSON file containing the training history.
                    Defaults to "history.json".
    Returns:
        dict: A dictionary containing the training history data loaded from the JSON file.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    return history


def plot_history_from_file(path="history.json", save_path="training_history.pdf"):
    """
    Plots training and validation curves (loss + accuracy) with epochs on the x-axis.
    Infers steps_per_epoch and validation interval automatically from history.
    """
    history = load_history(path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Infer steps_per_epoch from training history
    steps_per_epoch = max(history["train"]["batch"]) + 1
    val_steps_per_epoch = max(history["val"]["batch_train_idx"]) + 1

    # Infer validation interval from val history
    if len(history["val"]["batch_train_idx"]) > 1:
        diffs = np.diff(history["val"]["batch_train_idx"])
        val_interval = int(np.median(diffs))  # use median difference
    else:
        val_interval = steps_per_epoch  # fallback if only 1 val recorded

    # --- Training steps/epochs ---
    train_steps = np.arange(len(history["train"]["loss"]))
    train_epochs = train_steps / steps_per_epoch

    # --- Validation steps/epochs ---
    val_steps = np.arange(len(history["val"]["epoch"]))
    val_epochs = val_steps / val_steps_per_epoch

    # --- Create figure with 2 subplots ---
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(
        train_epochs,
        history["train"]["loss"],
        "b-",
        alpha=0.7,
        linewidth=1.5,
        label="Training",
    )
    ax.plot(
        train_epochs, history["train"]["loss"], "bx", markersize=5, markeredgewidth=1.5
    )

    ax.plot(
        val_epochs,
        history["val"]["loss"],
        "r-",
        alpha=0.7,
        linewidth=2,
        label="Validation",
    )
    ax.plot(val_epochs, history["val"]["loss"], "rx", markersize=7, markeredgewidth=2)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Loss\n({steps_per_epoch} steps/epoch, validation every {val_interval} steps)"
    )
    ax.legend()
    num_epochs = max(history["train"]["epoch"]) + 1
    ax.set_xlim(0, num_epochs)

    ax = axes[1]
    ax.plot(
        train_epochs,
        history["train"]["metrics"]["accuracy"],
        "b-",
        alpha=0.7,
        linewidth=1.5,
        label="Training",
    )
    ax.plot(
        train_epochs,
        history["train"]["metrics"]["accuracy"],
        "bx",
        markersize=5,
        markeredgewidth=1.5,
    )

    ax.plot(
        val_epochs,
        history["val"]["metrics"]["accuracy"],
        "r-",
        alpha=0.7,
        linewidth=2,
        label="Validation",
    )
    ax.plot(
        val_epochs,
        history["val"]["metrics"]["accuracy"],
        "rx",
        markersize=7,
        markeredgewidth=2,
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Accuracy\n({steps_per_epoch} steps/epoch, validation every {val_interval} steps)"
    )
    ax.legend()
    ax.set_xlim(0, num_epochs)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
