from config import BOARD_SIZE, categories, image_size
from tensorflow.keras import models
import numpy as np
import tensorflow as tf

class TicTacToePlayer:
    def get_move(self, board_state):
        raise NotImplementedError()

class UserInputPlayer:
    def get_move(self, board_state):
        inp = input('Enter x y:')
        try:
            x, y = inp.split()
            x, y = int(x), int(y)
            return x, y
        except Exception:
            return None

import random

class RandomPlayer:
    def get_move(self, board_state):
        positions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state[i][j] is None:
                    positions.append((i, j))
        return random.choice(positions)

from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2

class UserWebcamPlayer:

    def __init__(self):
        self.model = models.load_model(
            "results/optimized_model_25_epochs_timestamp_1785637417.keras"
        )

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

    def _process_frame(self, frame):
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )

        if len(faces) > 0:
            # Use the largest detected face.
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            # Keep more context around the face.
            margin_x = int(w * 0.20)
            margin_y = int(h * 0.20)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)

            x2 = min(
                gray.shape[1],
                x + w + margin_x
            )

            y2 = min(
                gray.shape[0],
                y + h + margin_y
            )

            gray = gray[
                y1:y2,
                x1:x2
            ]

        # Make it square.
        height, width = gray.shape
        size = min(height, width)

        start_y = (height - size) // 2
        start_x = (width - size) // 2

        gray = gray[
            start_y:start_y + size,
            start_x:start_x + size
        ]

        return gray
    def _access_webcam(self):
        cv2.namedWindow("preview")

        vc = cv2.VideoCapture(0)

        if not vc.isOpened():
            raise RuntimeError("Could not open webcam")

        final_frame = None

        while True:
            rval, frame = vc.read()

            if not rval:
                break

            # Show the original camera frame.
            # Do not crop every preview frame.
            cv2.imshow("preview", frame)

            key = cv2.waitKey(20)

            if key == 13:
                # Process only the frame captured when Enter is pressed.
                final_frame = self._process_frame(frame)
                break

        vc.release()
        cv2.destroyAllWindows()

        if final_frame is None:
            raise RuntimeError("Could not capture webcam frame")

        return final_frame

    def _print_reference(self, row_or_col):
        print('reference:')
        for i, emotion in enumerate(categories):
            print('{} {} is {}.'.format(row_or_col, i, emotion))
    
    def _get_row_or_col_by_text(self):
        try:
            val = int(input())
            return val
        except Exception as e:
            print('Invalid position')
            return None
    
    def _get_row_or_col(self, is_row):
        try:
            row_or_col = 'row' if is_row else 'col'
            self._print_reference(row_or_col)
            img = self._access_webcam()
            emotion = self._get_emotion(img)
            if type(emotion) is not int or emotion not in range(len(categories)):
                print('Invalid emotion number {}'.format(emotion))
                return None
            print('Emotion detected as {} ({} {}). Enter \'text\' to use text input instead (0, 1 or 2). Otherwise, press Enter to continue.'.format(categories[emotion], row_or_col, emotion))
            inp = input()
            if inp == 'text':
                return self._get_row_or_col_by_text()
            return emotion
        except Exception as e:
            # error accessing the webcam, or processing the image
            raise e
    
    def _get_emotion(self, img) -> int:
        img = cv2.resize(
            img,
            image_size
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

        img = np.expand_dims(
            img,
            axis=0
        )

        predictions = self.model.predict(
            img,
            verbose=0
        )


        return int(np.argmax(predictions[0]))

    def get_move(self, board_state):
        row, col = None, None
        while row is None:
            row = self._get_row_or_col(True)
        while col is None:
            col = self._get_row_or_col(False)
        return row, col
