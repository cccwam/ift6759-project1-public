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
