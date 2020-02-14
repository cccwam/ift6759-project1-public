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
    # Use only a maximum of 4 GPUs
    nb_gpus = tf.test.gpu_device_name()
    trainer_hyper_params = trainer_dict['hyper_params']
    print("Trainer hyper params:")
    print(trainer_hyper_params)
    print(valid_data_loader)

    mirrored_strategy = tf.distribute.MirroredStrategy(["/gpu:" + str(i) for i in range(min(1, len(nb_gpus)))])
    print("------------")
    print('Number of available GPU devices: {}'.format(nb_gpus))
    print('Number of used GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
    print("------------")

    # Create a unique id for the experiment for Tensorboard
    tensorboard_exp_id = helpers.get_tensorboard_experiment_id(
        experiment_name="dummy_model",
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )

    # Tensorboard logger for the different hyperparameters
    # TODO change this to the right hyperparameters space
    hp_optimizer = hp.HParam(
        'optimizer',
        hp.Discrete([
            "tf.keras.optimizers.Adam",
            "tf.keras.optimizers.SGD",
            "libs.custom.dummy_optimizer.MySGD_with_lower_learning_rate"
        ])
    )

    # Main loop to iterate over all possible hyperparameters
    variation_num = 0
    # TODO change this to the right hyperparameters space
    for optimizer in hp_optimizer.domain.values:
        hparams = {
            hp_optimizer: optimizer,
        }
        tensorboard_log_dir = os.path.join(tensorboard_exp_id, str(variation_num))
        print("Start variation id:", tensorboard_log_dir)
        train_test_model(
            dataset=train_data_loader,
            model=model,
            hp_optimizer=hp_optimizer,
            epochs=20,
            tensorboard_log_dir=tensorboard_log_dir,
            hparams=hparams,
            mirrored_strategy=mirrored_strategy
        )
        variation_num += 1
        break


def train_test_model(
        dataset,
        model,
        hp_optimizer,
        epochs,
        tensorboard_log_dir,
        hparams,
        mirrored_strategy,
        checkpoints_dir="/project/cq-training-1/project1/teams/team03/checkpoints"
):
    """
    Training loop

    :param dataset:
    :param model:
    :param hp_optimizer:
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
            compiled_model = helpers.compile_model(model, hparams, hp_optimizer)
    else:
        compiled_model = helpers.compile_model(model, hparams, hp_optimizer)

    callbacks = [
        # Workaround for https://github.com/tensorflow/tensorboard/issues/2412
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir), profile_batch=0),
        hp.KerasCallback(writer=str(tensorboard_log_dir), hparams=hparams),

        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_dir,
                                           save_weights_only=True),
    ]

    compiled_model.fit(dataset, epochs=epochs, callbacks=callbacks)


def train_simple(model, data_loader, tensorboard_tracking_folder,
                 validation_loader=None):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError())
    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001,
        patience=5)
    model.fit(data_loader,
              epochs=100,
              callbacks=[earlystop_callback],
              validation_data=validation_loader,
              shuffle=False)


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
