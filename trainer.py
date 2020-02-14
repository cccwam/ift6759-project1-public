# Summary:
#   Trains the predictor

import argparse
import typing
import os

# netCDF4 has to be imported before tensorflow because of hdf5 issues
import netCDF4
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from libs import helpers
# from tools.dummy_dataset_generator import generate_dummy_dataset

_ = netCDF4  # surpress unused module warning


def main(
        admin_config_path: typing.AnyStr,
        user_config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    admin_config_dict = helpers.load_dict(admin_config_path)
    user_config_dict = helpers.load_dict(user_config_path)
    validation_config_dict = helpers.load_dict(admin_config_path.replace('_train.json', '_validation.json'))
    trainer_dict = user_config_dict['trainer']

    helpers.validate_admin_config(admin_config_dict)
    helpers.validate_user_config(user_config_dict)

    train_data_loader = helpers.get_online_data_loader(user_config_dict, admin_config_dict)
    valid_data_loader = helpers.get_online_data_loader(user_config_dict, validation_config_dict, data_mode='validation')
    model = helpers.get_online_model(user_config_dict, admin_config_dict)

    train_model(model, trainer_dict, train_data_loader, valid_data_loader, tensorboard_tracking_folder)
    model.save(helpers.generate_model_name(user_config_dict))


def train_model(model, trainer_dict, train_data_loader, valid_data_loader, tensorboard_tracking_folder):
    # Activate this for multi gpu
    nb_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    trainer_hyper_params = trainer_dict['hyper_params']
    print("Trainer hyper params:")
    print(trainer_hyper_params)
    print(valid_data_loader)

    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(1, len(nb_gpus)))])
    print("------------")
    print('Number of available GPU devices: {}'.format(nb_gpus))
    print('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    print("------------")

    # Multi GPU setup
    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
        with mirrored_strategy.scope():
            model = helpers.get_online_model(user_config_dict, admin_config_dict)
    else:
        model = helpers.get_online_model(user_config_dict, admin_config_dict)

    train_model(model, train_data_loader, tensorboard_tracking_folder,
                mirrored_strategy=mirrored_strategy,
                 validation_dataset=valid_data_loader)


def train_model(model,
                data_loader,
                tensorboard_tracking_folder,
                mirrored_strategy,
                validation_dataset=None):


    # Create a unique id for the experiment for Tensorboard
    tensorboard_exp_id = helpers.get_tensorboard_experiment_id(
        experiment_name="dummy_model",
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )

    # Tensorboard logger for the different hyperparameters
    # Keep model and dataloader class names
    hp_model = hp.HParam('model_class', hp.Discrete([model.__class__.__name__]))
    hp_dataloader = hp.HParam('dataloader_class', hp.Discrete([data_loader.__class__.__name__]))
    # Hyperparameters search for the training loop
    hp_epochs = hp.HParam('epochs', hp.Discrete([100]))
    hp_learning_rate = hp.HParam('learning_rate', hp.Discrete([1e-3, 3e-3]))
    hp_patience = hp.HParam('patience', hp.Discrete([2]))

    data_loader = data_loader.batch(32)
    validation_dataset = validation_dataset.batch(32)

    # Main loop to iterate over all possible hyperparameters
    variation_num = 0
    for epochs in hp_epochs.domain.values:
        for learning_rate in hp_learning_rate.domain.values:
            for patience in hp_patience.domain.values:
                hparams = {
                    hp_model: hp_model.domain.values[0],
                    hp_dataloader: hp_dataloader.domain.values[0],
                    hp_epochs: epochs,
                    hp_learning_rate: learning_rate,
                    hp_patience: patience,
                }
                tensorboard_log_dir = os.path.join(tensorboard_exp_id, str(variation_num))
                print("Start variation id:", tensorboard_log_dir)
                train_test_model(
                    dataset=data_loader,
                    model=model,
                    tensorboard_log_dir=tensorboard_log_dir,
                    hparams=hparams,
                    mirrored_strategy=mirrored_strategy,
                    validation_dataset=validation_dataset,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    patience=patience
                )
                variation_num += 1
                break


def train_test_model(
        dataset,
        model,
        tensorboard_log_dir,
        hparams,
        mirrored_strategy,
        validation_dataset,
        epochs,
        learning_rate,
        patience,
    checkpoints_dir="/project/cq-training-1/project1/teams/team03/checkpoints"
):
    """
    Training loop

    :param dataset:
    :param model:
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
            compiled_model = helpers.compile_model(model, learning_rate=learning_rate)
    else:
        compiled_model = helpers.compile_model(model, learning_rate=learning_rate)

    callbacks = [
        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), profile_batch=0),
        hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams),
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                           save_weights_only=True),
    ]

    compiled_model.fit(dataset, epochs=epochs, callbacks=callbacks,
                       validation_data=validation_dataset)



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
