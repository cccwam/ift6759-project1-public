import json
import os
import pickle
import uuid
from datetime import datetime, timedelta

import jsonschema
import tensorflow as tf
import pvlib
import numpy as np
import pandas as pd


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
        preprocessed_data_path=None
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
    :param preprocessed_data_path: A path to the folder containing the preprocessed data
    :return: An instance of user_config_dict['model']['definition']['module'].['name']
    """
    if admin_config_dict:
        dataframe_path = admin_config_dict['dataframe_path']
        with open(dataframe_path, 'rb') as df_file_handler:
            dataframe = pickle.load(df_file_handler)
        target_datetimes = [
            datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%S')
            for date_time in admin_config_dict['target_datetimes']
        ]
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
        preprocessed_data_path=preprocessed_data_path
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
    default_model_path = '../model/best_model.hdf5'
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
    return "{}.{}.{}.hdf5".format(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name'],
        uuid.uuid4().hex
    )


def get_tensorboard_experiment_id(experiment_name, tensorboard_tracking_folder):
    """
    Create a unique id for TensorBoard for the experiment

    :param experiment_name: name of experiment
    :param tensorboard_tracking_folder: Path where to store TensorBoard data and save trained model
    """
    model_sub_folder = experiment_name + "-" + datetime.utcnow().isoformat()
    return os.path.join(tensorboard_tracking_folder, model_sub_folder)


def compile_model(model, learning_rate):
    """
        Helper function to compile a new model at each variation of the experiment
    :param learning_rate:
    :param model:
    :return:
    """

    model_instance = model

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model_instance.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return model_instance


def remove_night_values(dataframe):
    """Remove nighttime values from a GHI dataframe

    :param dataframe: pandas dataframe
    :return: pandas dataframe with nighttime values removes
    """

    return dataframe[(dataframe.DRA_DAYTIME == 1) |
                     (dataframe.TBL_DAYTIME == 1) |
                     (dataframe.BND_DAYTIME == 1) |
                     (dataframe.FPK_DAYTIME == 1) |
                     (dataframe.GWN_DAYTIME == 1) |
                     (dataframe.PSU_DAYTIME == 1) |
                     (dataframe.SXF_DAYTIME == 1)]


def remove_null_path(dataframe):
    """Remove entries with missing netcdf path

    :param dataframe: pandas dataframe
    :return: pandas dataframe with entries with missing netcdf path removed
    """

    return dataframe[dataframe['ncdf_path'] != 'nan']


# Since the first GHI values of the DRA station are NaN, it cannot
# inteprolate values, we will have to decide how to take of them
def fill_ghi(dataframe):
    stations = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']
    for station in stations:
        dataframe[f'{station}_GHI'] = (dataframe[f"{station}_GHI"]).interpolate(method='linear')

    return dataframe


def get_module_name(module_dictionary):
    return module_dictionary["definition"]["module"].split(".")[-1] + "." + module_dictionary["definition"]["name"]


def get_clearsky_predictions(
        stations,
        station_name,
        target_datetimes,
        target_time_offsets,
        index
):
    # Implementation of the clear sky model as described in pvlib
    # documentation at
    # https://pvlib-python.readthedocs.io/en/stable/clearsky.html
    latitude, longitude = stations[station_name][0], stations[station_name][1]
    altitude = stations[station_name][2]
    times = pd.date_range(
        start=target_datetimes[index].isoformat(),
        end=(target_datetimes[index] + target_time_offsets[3]).isoformat(),
        freq='1H'
    )
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    apparent_zenith = solpos['apparent_zenith']
    airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
    pressure = pvlib.atmosphere.alt2pres(altitude)
    airmass = pvlib.atmosphere.get_absolute_airmass(airmass,
                                                    pressure)
    linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(
        times, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    # an input is a pandas Series, so solis is a DataFrame
    ineichen = pvlib.clearsky.ineichen(
        apparent_zenith,
        airmass,
        linke_turbidity,
        altitude,
        dni_extra
    )
    return np.array([
            ineichen.ghi[0],
            ineichen.ghi[1],
            ineichen.ghi[3],
            ineichen.ghi[6]
        ],
        dtype=np.float32
    )


def get_mirrored_strategy():
    nb_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(2, nb_gpus))])
    print("------------")
    print('Number of available GPU devices: {}'.format(nb_gpus))
    print('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    print("------------")
    return mirrored_strategy
