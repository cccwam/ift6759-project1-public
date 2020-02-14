import json
import jsonschema
import uuid
from datetime import datetime, timedelta
import os
import sys
import pickle
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


def get_online_data_loader(
        user_config_dict,
        admin_config_dict=None,
        dataframe=None,
        target_datetimes=None,
        stations=None,
        target_time_offsets=None,
        data_mode='train'
):
    """
    Get an online version of the data loader defined in user_config_dict

    If admin_config_dict is not specified, the following have to be specified:
        * dataframe
        * target_datetimes
        * stations
        * target_time_offsets
    If admin_config_dict is specified, it overwrites the parameters specified above.

    :param user_config_dict: The user dictionary used to store user model/dataloader parameters
    :param admin_config_dict: The admin dictionary used to store train set parameters
    :param dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset) for all
            relevant timestamp values over the test period.
    :param target_datetimes: a list of timestamps that your data loader should use to provide imagery for your model.
            The ordering of this list is important, as each element corresponds to a sequence of GHI values
            to predict. By definition, the GHI values must be provided for the offsets given by ``target_time_offsets``
            which are added to each timestamp (T=0) in this datetimes list.
    :param stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :return: An instance of user_config_dict['model']['definition']['module'].['name']
    """
    if admin_config_dict:
        dataframe_path = admin_config_dict['dataframe_path']
        with open(dataframe_path, 'rb') as df_file_handler:
            dataframe = pickle.load(df_file_handler)
        target_datetimes = [datetime.strptime(s, '%Y-%m-%dT%H:%M:%S') for s in admin_config_dict['target_datetimes']]
        stations = admin_config_dict['stations']
        target_time_offsets = [timedelta(hours=h) for h in [0, 1, 3, 6]]  # hard coded

    return import_from(
        user_config_dict['data_loader']['definition']['module'],
        user_config_dict['data_loader']['definition']['name']
    )(
        dataframe=dataframe,
        target_datetimes=target_datetimes,
        stations=stations,
        target_time_offsets=target_time_offsets,
        config=user_config_dict,
        data_mode=data_mode
    )


def get_online_model(
        user_config_dict,
        admin_config_dict=None,
        stations=None,
        target_time_offsets=None,
):
    """
    Get an online version of the model defined in user_config_dict

    If admin_config_dict is not specified, the following have to be specified:
        * stations
        * target_time_offsets
    If admin_config_dict is specified, it overwrites the parameters specified above.

    :param user_config_dict: The user dictionary used to store user model/dataloader parameters
    :param admin_config_dict: The admin dictionary used to store train set parameters
    :param stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
    :param target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
    :return: An instance of user_config_dict['model']['definition']['module'].['name']
    """
    if admin_config_dict:
        stations = admin_config_dict['stations'],
        target_time_offsets = [timedelta(hours=h) for h in [0, 1, 3, 6]]  # hard coded

    return import_from(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name']
    )(
        stations=stations,
        target_time_offsets=target_time_offsets,
        config=user_config_dict
    )


def prepare_model(
        user_config_dict,
        stations,
        target_time_offsets
):
    """
    Prepare model

    Args:
        user_config_dict: configuration dictionary holding any extra parameters that might be required by the user.
            These parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).

    Returns:
        A ``tf.keras.Model`` object that can be used to generate new GHI predictions given imagery tensors.
    """
    default_model_path = '../model/best_model.h5'
    model_source = user_config_dict['model']['source']

    if model_source == 'online':
        return get_online_model(
            user_config_dict=user_config_dict,
            stations=stations,
            target_time_offsets=target_time_offsets
        )
    elif model_source:
        if not os.path.exists(model_source):
            raise FileNotFoundError(f'Error: The file {model_source} does not exist.')
    else:
        if os.path.exists(default_model_path):
            model_source = default_model_path
        else:
            raise FileNotFoundError(f'Error: The file {default_model_path} does not exist.')

    return tf.keras.models.load_model(model_source)


def generate_model_name(user_config_dict):
    return "{}.{}.{}.tf".format(
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

    model_instance = model

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model_instance
