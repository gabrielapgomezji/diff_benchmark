import json
import os

import matplotlib.pyplot as plt


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

    with open(path, "r") as f:
        history = json.load(f)
    return history


def plot_history_from_file(path="history.json", save_path="training_history.pdf"):
    """
    Plots training and validation loss and accuracy from a JSON history file.
    Parameters:
        path (str): The file path to the JSON file containing training history data.
                    Default is "history.json".
        save_path (str): The file path where the plot image will be saved.
                         Default is "training_history.png".
    Returns:
        None
    This function loads the training history from the specified JSON file,
    creates a plot with two subplots (one for loss and one for accuracy),
    and saves the resulting figure to the specified save path.
    """

    history = load_history(path)

    # Ensure the parent directory of save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))

    # --- Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(history["train"]["loss"], label="Train Loss", alpha=0.7)
    plt.plot(history["val"]["loss"], label="Val Loss", alpha=0.7)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over training")

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(history["train"]["accuracy"], label="Train Acc", alpha=0.7)
    plt.plot(history["val"]["accuracy"], label="Val Acc", alpha=0.7)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over training")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
