import os
import json

from jsonschema import validate


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def load_user_config(user_config_path):
    assert os.path.isfile(user_config_path), f'User config file not found: {user_config_path}'
    user_config_file = load_json_file(user_config_path)
    user_schema_file = load_json_file('configs/user/schema.json')
    validate(user_config_file, user_schema_file)
    return user_config_file


def load_admin_config(admin_config_path):
    assert os.path.isfile(admin_config_path), f'Admin config file not found: {admin_config_path}'
    admin_config_file = load_json_file(admin_config_path)
    admin_schema_file = load_json_file('configs/admin/schema.json')
    validate(admin_config_file, admin_schema_file)
    return admin_config_file


def load_json_file(file_path):
    with open(file_path, 'r') as file_handler:
        file_data = file_handler.read()
    return json.loads(file_data)


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
