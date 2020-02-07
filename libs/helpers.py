import json
import jsonschema
import uuid
from datetime import datetime
import os
import sys
from importlib import import_module

import tensorflow as tf


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
