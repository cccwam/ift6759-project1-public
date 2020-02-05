# Summary:
#   Trains the predictor

import argparse
import typing

from libs import utils


def main(
        admin_config_path: typing.AnyStr,
        user_config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    admin_config_dict = utils.load_admin_config(admin_config_path)
    user_config_dict = utils.load_user_config(user_config_path)

    data_loader = utils.get_data_loader(admin_config_dict, user_config_dict)
    model = utils.get_model(admin_config_dict, user_config_dict)

    # TODO: Implement a similar signature as the following
    model.train(data_loader, tensorboard_tracking_folder)


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
