from tensorflow.keras.metrics import Metric
import tensorflow as tf
import numpy as np


class ClassAccuracy(Metric):

    def __init__(self, name='class_accuracy', **kwargs):
        super(ClassAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        K = len(set(y_true))
        yp = tf.argmax(y_pred, axis=1)
        acc = []
        for i in range(K):
            a = np.mean((yp[y_true == i] == y_true[y_true == i]).numpy())
            acc.append(a)
        accuracies = tf.convert_to_tensor(acc)
        self.accuracies = accuracies

    def result(self):
        return self.accuracies
