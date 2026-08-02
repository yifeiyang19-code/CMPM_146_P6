import os
import random
import shutil


SOURCE_TRAIN = "kaggle/train"
SOURCE_TEST = "kaggle/test"

TARGET_TRAIN = "transfer_train"
TARGET_TEST = "transfer_test"

CATEGORIES = ["angry", "fear", "sad"]

TRAIN_PER_CATEGORY = {
    "angry": 1666,
    "fear": 1666,
    "sad": 1668
}

RANDOM_SEED = 47


def reset_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.makedirs(directory)


def copy_random_files(source, destination, count=None):
    files = [
        file_name
        for file_name in os.listdir(source)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(files)

    if count is not None:
        files = files[:count]

    for file_name in files:
        source_path = os.path.join(
            source,
            file_name
        )

        destination_path = os.path.join(
            destination,
            file_name
        )

        shutil.copy2(
            source_path,
            destination_path
        )

    return len(files)


def main():
    random.seed(RANDOM_SEED)

    reset_directory(TARGET_TRAIN)
    reset_directory(TARGET_TEST)

    for category in CATEGORIES:
        train_source = os.path.join(
            SOURCE_TRAIN,
            category
        )

        test_source = os.path.join(
            SOURCE_TEST,
            category
        )

        train_destination = os.path.join(
            TARGET_TRAIN,
            category
        )

        test_destination = os.path.join(
            TARGET_TEST,
            category
        )

        os.makedirs(train_destination)
        os.makedirs(test_destination)

        train_count = copy_random_files(
            train_source,
            train_destination,
            TRAIN_PER_CATEGORY[category]
        )

        test_count = copy_random_files(
            test_source,
            test_destination
        )

        print(
            category,
            "train:",
            train_count,
            "test:",
            test_count
        )


if __name__ == "__main__":
    main()