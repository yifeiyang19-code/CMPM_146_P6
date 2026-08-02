from models.model import Model

from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import RMSprop


BASIC_MODEL_PATH = (
    "results/"
    "basic_model_10_epochs_timestamp_1785476957.keras"
)


class TransferedModel(Model):
    def _define_model(
        self,
        input_shape,
        categories_count
    ):
        loaded_model = load_model(
            BASIC_MODEL_PATH
        )

        feature_extractor = KerasModel(
            inputs=loaded_model.input,
            outputs=loaded_model.layers[-3].output
        )

        feature_extractor.trainable = False

        self.model = Sequential([
            feature_extractor,

            layers.Flatten(),

            layers.Dropout(0.20),

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