"""
    Dummy dataset. Only used as we don't have yet the full dataloader.
    inspired by https://keras.io/examples/conv_lstm/

"""
import argparse
import pickle
import time

import numpy as np
import tensorflow as tf


def generate_dummy_dataset(batch_size, n_samples=1200, n_frames=15, frame_shape=(80, 80)):
    """
    Generate a dummy dataset of movies of random noise

    :param batch_size: The amount of entries the dataset will return as a batch
    :param n_samples: The number of movies
    :param n_frames: Each movie's number of frames
    :param frame_shape: The shape of each frame
    :return: tf.data.Dataset
    """

    noisy_movies, labels = generate_noisy_movies(n_samples, n_frames, frame_shape)

    # TODO change the input here
    dataset = tf.data.Dataset.from_tensor_slices((noisy_movies[:100], labels[:100]))

    # TODO split dataset into train validation

    # for i, (imgs, labels) in enumerate(dataset):
    #    print(i, labels)

    # TODO review batch
    dataset = dataset.batch(batch_size)

    # TODO activate cache ?
    # TODO activate prefect
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Simulate slow opening of file
    def slow_process(imgs, labels):
        time.sleep(1)
        return imgs, labels * 2

    return dataset.map(slow_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def generate_noisy_movies(n_samples=1200, n_frames=15, frame_shape=(80, 80)):
    """
    Generate noisy movies

    :param n_samples: The number of movies
    :param n_frames: Each movie's number of frames
    :param frame_shape: The shape of each frame
    :return: noisy_movies: np.array, labels: np.array
    """
    noisy_movies = np.zeros(
        (n_samples, n_frames, frame_shape[0], frame_shape[1], 1),
        dtype=np.float
    )

    labels = []
    for i in range(n_samples):
        # Initial position
        x_start = np.random.randint(20, 60)
        y_start = np.random.randint(20, 60)
        # Direction of motion
        direction_x = np.random.randint(0, 3) - 1
        direction_y = np.random.randint(0, 3) - 1

        # Size of the square
        w = np.random.randint(2, 4)

        for t in range(n_frames):
            x_shift = x_start + direction_x * t
            y_shift = y_start + direction_y * t
            noisy_movies[i, t, x_shift - w: x_shift + w, y_shift - w: y_shift + w, 0] += 1

            # Make it more robust by adding noise.
            # The idea is that if during inference,
            # the value of the pixel is not exactly one,
            # we need to train the network to be robust and still
            # consider it as a pixel belonging to a square.
            if np.random.randint(0, 2):
                noise_f = (-1) ** np.random.randint(0, 2)
                dl = w + 1
                noisy_movies[i, t, x_shift - dl: x_shift + dl, y_shift - dl: y_shift + dl, 0] += noise_f * 0.1
        labels += [[direction_x, direction_y, x_start, y_start]]

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1

    labels = np.array(labels)

    return noisy_movies, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int,
                        help="The amount of entries the dataset will return as a batch")
    parser.add_argument("-ns", "--n_samples", type=int,
                        help="The number of movies")
    parser.add_argument("-nf", "--n_frames", type=int, default=None,
                        help="Each movie's number of frames")
    parser.add_argument("-fw", "--frame_width", type=int, default=None,
                        help="The width of each frame")
    parser.add_argument("-fh", "--frame_height", type=int, default=None,
                        help="The height of each frame")
    parser.add_argument("-sf", "--pickle_save_file", type=str, default=None,
                        help="The pickle file to save the dataset to")
    args = parser.parse_args()
    dataset = generate_dummy_dataset(
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        n_frames=args.n_frames,
        frame_shape=(args.frame_width, args.frame_height)
    )
    pickle.dump(dataset, open(args.pickle_save_file, 'wb'))
