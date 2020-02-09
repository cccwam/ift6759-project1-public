# Summary:
#   Trains the predictor

import argparse
import typing

from libs import helpers


def main(
        admin_config_path: typing.AnyStr,
        user_config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    admin_config_dict = helpers.load_dict(admin_config_path)
    user_config_dict = helpers.load_dict(user_config_path)

    helpers.validate_admin_config(admin_config_dict)
    helpers.validate_user_config(user_config_dict)

    data_loader = helpers.get_online_data_loader(admin_config_dict, user_config_dict)
    model = helpers.get_online_model(admin_config_dict, user_config_dict)

    model.train(data_loader, tensorboard_tracking_folder)
    model.save(helpers.generate_model_name(user_config_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--admin_cfg_path', type=str,
                        help='path to the JSON config file used to store train set parameters')
    parser.add_argument('-u', '--user_cfg_path', type=str, default=None,
                        help='path to the JSON config file used to store user model/dataloader parameters')
    parser.add_argument('-t', '--tensorboard_tracking_folder', type=str, default=None,
                        help='path where to store tensorboard data and save trained model')
    args = parser.parse_args()
    main(
        admin_config_path=args.admin_cfg_path,
        user_config_path=args.user_cfg_path,
        tensorboard_tracking_folder=args.tensorboard_tracking_folder
    )
