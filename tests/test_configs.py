import jsonschema
import os

from libs import helpers


def validate_configs(configs_folder):
    schema = helpers.load_dict(os.path.join(configs_folder, 'schema.json'))
    for file_name in os.listdir(configs_folder):
        if file_name == 'schema.json':
            continue
        config_file_path = os.path.join(configs_folder, file_name)
        try:
            jsonschema.validate(
                schema=schema,
                instance=helpers.load_dict(config_file_path)
            )
        except jsonschema.exceptions.ValidationError as e:
            print(f"ValidationError for the file {config_file_path}")
            raise e


def test_validate_admin_configs():
    """
    Validates that admin configuration files adhere to the admin configs schema
    """
    validate_configs('configs/admin')


def test_validate_user_configs():
    """
    Validates that user configuration files adhere to the user configs schema
    """
    validate_configs('configs/user')
