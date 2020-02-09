import json
import jsonschema
import uuid


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


def get_data_loader(
        user_config_dict,
        admin_config_dict=None,
        dataframe=None,
        target_datetimes=None,
        stations=None,
        target_time_offsets=None,
):
    """
    Get data loader

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
        dataframe = admin_config_dict['dataframe_path']
        target_datetimes = admin_config_dict['target_datetimes']
        stations = admin_config_dict['stations'],
        target_time_offsets = admin_config_dict['target_time_offsets']

    return import_from(
        user_config_dict['data_loader']['definition']['module'],
        user_config_dict['data_loader']['definition']['name']
    )(
        dataframe=dataframe,
        target_datetimes=target_datetimes,
        stations=stations,
        target_time_offsets=target_time_offsets,
        config=user_config_dict
    )


def get_model(
        user_config_dict,
        admin_config_dict=None,
        stations=None,
        target_time_offsets=None,
):
    """
    Get model

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
        target_time_offsets = admin_config_dict['target_time_offsets']

    return import_from(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name']
    )(
        stations=stations,
        target_time_offsets=target_time_offsets,
        config=user_config_dict
    )


def generate_model_name(user_config_dict):
    return "{}.{}.{}.h5".format(
        user_config_dict['model']['definition']['module'],
        user_config_dict['model']['definition']['name'],
        uuid.uuid4().hex
    )
