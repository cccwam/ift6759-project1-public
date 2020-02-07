import json
import jsonschema
import uuid
from datetime import datetime
import os
from importlib import import_module
import sys

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tools.dummy_dataset_generator import generate_dummy_dataset


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def load_dict(json_file_path):
    with open(json_file_path, 'r') as file_handler:
        file_data = file_handler.read()
    return json.loads(file_data)


def validate_admin_config(admin_config_dict):
    schema_file = load_dict('configs/admin/schema.json')
    jsonschema.validate(admin_config_dict, schema_file)


def validate_user_config(user_config_dict):
    schema_file = load_dict('configs/user/schema.json')
    jsonschema.validate(user_config_dict, schema_file)


def get_data_loader(admin_config_dict, user_config_dict):
    return import_from(
        user_config_dict['data_loader']['definition']['module'],
        user_config_dict['data_loader']['definition']['name']
    )(
        dataframe=admin_config_dict['dataframe_path'],
        target_datetimes=admin_config_dict['target_datetimes'],
        stations=admin_config_dict['stations'],
        target_time_offsets=admin_config_dict['target_time_offsets'],
        config=user_config_dict
    )


def get_model(admin_config_dict, user_config_dict):
    return import_from(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name']
    )(
        stations=admin_config_dict['stations'],
        target_time_offsets=admin_config_dict['target_time_offsets'],
        config=user_config_dict
    )


def generate_model_name(user_config_dict):
    return "{}.{}.{}.h5".format(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name'],
        uuid.uuid4().hex
    )


def get_tensorboard_experiment_id(experiment_name, tensorboard_tracking_folder):
    """
    Create a unique id for tensorboard for the experiment

    :param experiment_name: name of experiment
    :param tensorboard_tracking_folder: Path where to store tensorboard data and save trained model
    """
    model_sub_folder = experiment_name + "-" + datetime.utcnow().isoformat()
    return os.path.join(tensorboard_tracking_folder, model_sub_folder)


def train_model(model, data_loader, tensorboard_tracking_folder):
    # Activate this for multi gpu
    # Use only a maximum of 4 GPUs
    nb_gpus = tf.test.gpu_device_name()

    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(2, len(nb_gpus)))])
    print("------------")
    print('Number of available GPU devices: {}'.format(nb_gpus))
    print('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    print("------------")

    # Create a unique id for the experiment for Tensorboard
    tensorboard_exp_id = get_tensorboard_experiment_id(
        experiment_name="dummy_model",
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )

    # Tensorboard logger for the different hyperparameters
    # TODO change this to the right hyperparameters space
    hp_optimizer = hp.HParam(
        'optimizer',
        hp.Discrete([
            "tf.keras.optimizers.Adam",
            "tf.keras.optimizers.SGD",
            "libs.custom.dummy_optimizer.MySGD_with_lower_learning_rate"
        ])
    )

    # Main loop to iterate over all possible hyperparameters
    variation_num = 0
    # TODO change this to the right hyperparameters space
    for optimizer in hp_optimizer.domain.values:
        hparams = {
            hp_optimizer: optimizer,
        }
        tensorboard_log_dir = os.path.join(tensorboard_exp_id, str(variation_num))
        print("Start variation id:", tensorboard_log_dir)
        train_test_model(
            dataset=generate_dummy_dataset(batch_size=16),  # TODO change this for the right dataset
            model=model,
            hp_optimizer=hp_optimizer,
            epochs=5,
            tensorboard_log_dir=tensorboard_log_dir,
            hparams=hparams,
            mirrored_strategy=mirrored_strategy
        )
        variation_num += 1


def compile_model(model, hparams, hp_optimizer):
    """
        Helper function to compile a new model at each variation of the experiment
    :param model:
    :param hparams:
    :param hp_optimizer:
    :return:
    """

    model_instance = model()

    # Workaround to get the right optimizer from class path
    # Because hparams only accept dtype string not class
    # See https://stackoverflow.com/questions/3451779/how-to-dynamically-create-an-instance-of-a-class-in-python
    class_name = hparams[hp_optimizer].rsplit('.', 1)
    if len(class_name) > 1:
        module_path, class_name = class_name
        module_path = module_path.replace("tf", "tensorflow")
        module = import_module(module_path)
    else:
        class_name = class_name[0]
        module = sys.modules[__name__]
    optimizer_instance = getattr(module, class_name)()

    model_instance.compile(
        optimizer=optimizer_instance,
        loss=tf.keras.losses.mean_squared_error,
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model_instance


def train_test_model(
        dataset,
        model,
        hp_optimizer,
        epochs,
        tensorboard_log_dir,
        hparams,
        mirrored_strategy,
        checkpoints_dir="/project/cq-training-1/project1/teams/team03/checkpoints"
):
    """
    Training loop

    :param dataset:
    :param model:
    :param hp_optimizer:
    :param epochs:
    :param tensorboard_log_dir:
    :param hparams:
    :param mirrored_strategy:
    :param checkpoints_dir:
    :return:
    """
    # Multi GPU setup
    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            compiled_model = compile_model(model, hparams, hp_optimizer)
    else:
        compiled_model = compile_model(model, hparams, hp_optimizer)

    callbacks = [
        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), profile_batch=0),
        hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams),

        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                           save_weights_only=True),
    ]

    compiled_model.fit(dataset, epochs=epochs, callbacks=callbacks)
