# Summary:
#   Trains the predictor

import tensorflow as tf


def main():
    # TODO: Implement

    tf.debugging.set_log_device_placement(True)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Running a matrix multiplication using two tensors...")

    # Place some tensors on the GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    c = tf.matmul(a, b)
    print(c)


if __name__ == '__main__':
    main()
