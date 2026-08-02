from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size

import matplotlib.pyplot as plt


input_shape = (
    image_size[0],
    image_size[1],
    3
)

categories_count = 3

models = {
    "transfered_model": TransferedModel,
    "random_model": RandomModel
}


def plot_history_diff(
    transfered_history,
    random_history
):
    transfered_accuracy = (
        transfered_history.history[
            "val_accuracy"
        ]
    )

    random_accuracy = (
        random_history.history[
            "val_accuracy"
        ]
    )

    transfered_epochs = range(
        1,
        len(transfered_accuracy) + 1
    )

    random_epochs = range(
        1,
        len(random_accuracy) + 1
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        transfered_epochs,
        transfered_accuracy,
        label="Transferred Model Accuracy"
    )

    plt.plot(
        random_epochs,
        random_accuracy,
        label="Random Model Accuracy"
    )

    plt.grid(True)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Transferred Features vs Random Features"
    )

    plt.show()


if __name__ == "__main__":
    epochs = 15

    print("* Data preprocessing")

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = get_transfer_datasets()

    histories = []

    for name, model_class in models.items():
        print(
            "* Training {} for {} epochs".format(
                name,
                epochs
            )
        )

        model = model_class(
            input_shape,
            categories_count
        )

        model.print_summary()

        history = model.train_model(
            train_dataset,
            validation_dataset,
            epochs
        )

        histories.append(history)

        print(
            "* Evaluating {}".format(name)
        )

        model.evaluate(test_dataset)

        print(
            "* Confusion Matrix for {}".format(
                name
            )
        )

        print(
            model.get_confusion_matrix(
                test_dataset
            )
        )

    plot_history_diff(
        histories[0],
        histories[1]
    )