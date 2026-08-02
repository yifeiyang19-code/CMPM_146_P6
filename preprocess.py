from keras.utils import image_dataset_from_directory

from config import (
    train_directory,
    test_directory,
    image_size,
    batch_size,
    validation_split,
    categories
)


TRANSFER_TRAIN_DIRECTORY = "transfer_train"
TRANSFER_TEST_DIRECTORY = "transfer_test"


def _split_data(
    train_directory,
    test_directory,
    batch_size,
    validation_split,
    class_names=None
):
    print('train dataset:')

    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset='both',
        seed=47,
        class_names=class_names
    )

    print('test dataset:')

    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        class_names=class_names
    )

    return train_dataset, validation_dataset, test_dataset


def get_datasets():
    return _split_data(
        train_directory,
        test_directory,
        batch_size,
        validation_split
    )


def get_transfer_datasets():
    return _split_data(
        TRANSFER_TRAIN_DIRECTORY,
        TRANSFER_TEST_DIRECTORY,
        batch_size,
        validation_split
    )