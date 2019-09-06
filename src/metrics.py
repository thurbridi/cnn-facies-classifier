import numpy as np
from keras.callbacks import Callback
import keras.backend as K
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)


class ConfusionMatrixCallback(Callback):
    def on_test_end(self, logs=None):
        print("confusion matrix on test end")


class F1ScoreCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = K.cast(K.argmax(self.model.outputs, axis=-1), K.floatx())
        y_true = np.asarray(K.argmax(self.model.targets,
                                     axis=-1), K.floatx())

        print(y_pred[0])
        print(y_true[0])

        _precision = precision_score(y_true, y_pred, average='macro')
        _recall = recall_score(y_true, y_pred, average='macro')
        _f1 = f1_score(y_true, y_pred, average='macro')

        print(f'\nPrecision:')
        print(f'validation (cur: {_precision})')

        print(f'\nRecall:')
        print(f'validation (cur: {_recall})')

        print(f'\nF1-Score:')
        print(f'validation (cur: {_f1})')

    def on_test_end(self, logs=None):
        pass
