from models.model import Model

from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping


class OptimizedModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(
                1.0 / 255,
                input_shape=input_shape
            ),

            layers.Conv2D(
                16,
                kernel_size=(3, 3),
                activation="relu"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu"
            ),
            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Dropout(0.15),

            layers.Flatten(),

            layers.Dense(
                categories_count,
                activation="softmax"
            )
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=RMSprop(
                learning_rate=0.0005
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train_model(
        self,
        train_dataset,
        validation_dataset,
        epochs
    ):
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )

        history = self.model.fit(
            x=train_dataset,
            epochs=epochs,
            verbose="auto",
            validation_data=validation_dataset,
            callbacks=[early_stopping]
        )

        return history