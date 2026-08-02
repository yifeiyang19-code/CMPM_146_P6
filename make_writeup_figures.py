import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


os.makedirs("writeup_figures", exist_ok=True)


def save_history(history_file, prefix):
    loaded = np.load(
        history_file,
        allow_pickle=True
    ).item()

    if hasattr(loaded, "history"):
        history = loaded.history
    else:
        history = loaded

    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(history["accuracy"]) + 1),
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        range(1, len(history["val_accuracy"]) + 1),
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        prefix.replace("_", " ").title()
        + " Accuracy"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "writeup_figures/"
        + prefix
        + "_accuracy.png",
        dpi=200
    )

    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(history["loss"]) + 1),
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        range(1, len(history["val_loss"]) + 1),
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        prefix.replace("_", " ").title()
        + " Loss"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "writeup_figures/"
        + prefix
        + "_loss.png",
        dpi=200
    )

    plt.close()


def save_summary(model_file, output_file):
    model = load_model(model_file)

    lines = []

    model.summary(
        print_fn=lambda x: lines.append(x)
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


# Initial model
save_history(
    "results/basic_model_10_epochs_timestamp_1785476957.npy",
    "initial_model"
)

save_summary(
    "results/basic_model_10_epochs_timestamp_1785476957.keras",
    "writeup_figures/initial_model_summary.txt"
)


# Optimized model
save_history(
    "results/optimized_model_25_epochs_timestamp_1785637417.npy",
    "optimized_model"
)

save_summary(
    "results/optimized_model_25_epochs_timestamp_1785637417.keras",
    "writeup_figures/optimized_model_summary.txt"
)

def summary_to_image(txt_file, output_file, title):
    with open(
        txt_file,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    lines = text.splitlines()

    height = max(6, len(lines) * 0.28)

    plt.figure(
        figsize=(12, height)
    )

    plt.axis("off")

    plt.title(
        title,
        fontsize=16,
        pad=20
    )

    plt.text(
        0.01,
        0.98,
        text,
        family="monospace",
        fontsize=9,
        verticalalignment="top"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


summary_to_image(
    "writeup_figures/initial_model_summary.txt",
    "writeup_figures/initial_model_summary.png",
    "Initial Model Architecture"
)

summary_to_image(
    "writeup_figures/optimized_model_summary.txt",
    "writeup_figures/optimized_model_summary.png",
    "Optimized Model Architecture"
)

print("Writeup figures created.")