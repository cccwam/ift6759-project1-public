"""
    @Author François Mercier


"""
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from libs.models.convlstm import my_conv_lstm_model_builder
from libs.helper.tensorboard_helper import tensorboard_experiment_id
from libs.datasets.dummy_dataset import helper_dummy_dataset

from importlib import import_module
import sys


def compile_model(model_builder, hparams):
    """
        Helper function to compile a new model at each variation of the experiment
    :param model_builder:
    :param hparams:
    :return:
    """

    model = model_builder()

    # Workaround to get the right optimizer from class path
    # Because hparams only accept dtype string not class
    # See https://stackoverflow.com/questions/3451779/how-to-dynamically-create-an-instance-of-a-class-in-python
    class_name = hparams[HP_OPTIMIZER].rsplit('.', 1)
    if len(class_name) > 1:
        module_path, class_name = class_name
        module_path = module_path.replace("tf", "tensorflow")
        module = import_module(module_path)
    else:
        class_name = class_name[0]
        module = sys.modules[__name__]
    optimizer_instance = getattr(module, class_name)()

    model.compile(optimizer=optimizer_instance,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_test_model(dataset, model_builder, epochs, tensorboard_log_dir, hparams, mirrored_strategy,
                     checkpoints_dir="/project/cq-training-1/project1/teams/team03/checkpoints"):
    """
        Training loop
    :param model_builder:
    :param epochs:
    :param mirrored_strategy:
    :param checkpoints_dir:
    :param dataset:
    :param tensorboard_log_dir:
    :param hparams:
    :return:
    """

    # Multi GPU setup
    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            model = compile_model(model_builder=model_builder, hparams=hparams)
    else:
        model = compile_model(model_builder=model_builder, hparams=hparams)

    callbacks = [
        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), profile_batch=0),
        hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams),

        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                           save_weights_only=True),
    ]

    model.fit(dataset, epochs=epochs, callbacks=callbacks)


# Activate this for multi gpu
# Use only a maximum of 4 GPUs
nb_gpus = tf.test.gpu_device_name()

mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(2, len(nb_gpus)))])
print("------------")
print('Number of available GPU devices: {}'.format(nb_gpus))
print('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
print("------------")

# Create a unique id for the experiment for Tensorboard
tensorboard_exp_id = tensorboard_experiment_id(
    initial="FM",  # TODO Change this to your own
    experiment_name="dummy_model")

# Tensorboard logger for the different hyperparameters
# TODO change this to the right hyperparameters space
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(["tf.keras.optimizers.Adam", "tf.keras.optimizers.SGD",
                                                   "libs.custom.dummy_optimizer.MySGD_with_lower_learning_rate"]))

# Main loop to iterate over all possible hyperparameters
variation_num = 0
# TODO change this to the right hyperparameters space
for optimizer in HP_OPTIMIZER.domain.values:
    hparams = {
        HP_OPTIMIZER: optimizer,
    }
    print("Start variation id:", tensorboard_exp_id / str(variation_num))
    train_test_model(dataset=helper_dummy_dataset(batch_size=16),  # TODO change this for the right dataset
                     model_builder=my_conv_lstm_model_builder,  # TODO change this to your own model
                     epochs=5,
                     tensorboard_log_dir=tensorboard_exp_id / str(variation_num), hparams=hparams,
                     mirrored_strategy=mirrored_strategy)
    variation_num += 1
