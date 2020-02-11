import tensorflow as tf


class MySGD_with_lower_learning_rate(tf.keras.optimizers.SGD):
    """
        Dummy class: only to illustrate how to compare Adam with no learning rate and SGD with diff learning rates
        NOT TO USE
    """
    def __index__(self):
        super(MySGD_with_lower_learning_rate, self).__init__(learning_rate=0.001)
