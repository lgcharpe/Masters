from tensorflow.keras.metrics import Metric
import tensorflow as tf


class ClassAccuracy(Metric):

    def __init__(self, name='class_accuracy', **kwargs):
        super(ClassAccuracy, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        K = len(set(y_true))
        oyp = tf.one_hot(tf.argmax(y_pred, axis=1), K)
        oyt = tf.one_hot(y_true, K)
        accuracies = tf.reduce_mean(tf.cast(oyp == oyt, tf.float32), axis=0)
        self.accuracies = accuracies

    def result(self):
        return self.accuracies
