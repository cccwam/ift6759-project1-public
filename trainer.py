# Summary:
#   Trains the predictor

import argparse
import os
import typing

# netCDF4 has to be imported before tensorflow because of hdf5 issues
import netCDF4
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from libs import helpers

_ = netCDF4  # surpress unused module warning


def main(
        training_config_path: typing.AnyStr,
        validation_config_path: typing.AnyStr,
        user_config_path: typing.AnyStr,
        tensorboard_tracking_folder: typing.AnyStr
):
    """
    Train a model

    :param training_config_path: path to the JSON config file used to store training set parameters
    :param validation_config_path: path to the JSON config file used to store validation set parameters
    :param user_config_path: path to the JSON config file used to store user model, dataloader and trainer parameters
    :param tensorboard_tracking_folder: path where to store TensorBoard data and save trained model
    """
    training_config_dict = helpers.load_dict(training_config_path)
    validation_config_dict = helpers.load_dict(validation_config_path)
    user_config_dict = helpers.load_dict(user_config_path)

    helpers.validate_admin_config(training_config_dict)
    helpers.validate_admin_config(validation_config_dict)
    helpers.validate_user_config(user_config_dict)

    training_source = user_config_dict['data_loader']['hyper_params']['preprocessed_data_source']['training']
    validation_source = user_config_dict['data_loader']['hyper_params']['preprocessed_data_source']['validation']

    training_data_loader = helpers.get_online_data_loader(
        user_config_dict, training_config_dict, preprocessed_data_path=training_source)
    validation_data_loader = helpers.get_online_data_loader(
        user_config_dict, validation_config_dict, preprocessed_data_path=validation_source)

    print("Eager mode", tf.executing_eagerly())

    mirrored_strategy = helpers.get_mirrored_strategy()

    train_models(
        user_config_dict=user_config_dict,
        training_config_dict=training_config_dict,
        training_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader,
        tensorboard_tracking_folder=tensorboard_tracking_folder,
        mirrored_strategy=mirrored_strategy
    )


def train_models(
        user_config_dict,
        training_config_dict,
        training_data_loader,
        validation_data_loader,
        tensorboard_tracking_folder,
        mirrored_strategy
):
    """
    Train the possible combinations of models based on the hyper parameters defined in  user_config_dict

    :param user_config_dict: A dictionary of the user configuration json file which contains the hyper parameters
    :param training_config_dict: A dictionary of the admin configuration json file which references the training data
    :param training_data_loader: The training data loader
    :param validation_data_loader: The validation data loader
    :param tensorboard_tracking_folder: The TensorBoard tracking folder
    :param mirrored_strategy: A tf.distribute.MirroredStrategy on how many GPUs to use during training
    """
    model_dict = user_config_dict['model']
    data_loader_dict = user_config_dict['data_loader']
    trainer_hyper_params = user_config_dict['trainer']['hyper_params']

    print(f"\nModel definitions: {model_dict}\n")
    print(f"\nData loader definitions: {data_loader_dict}\n")
    print(f"\nTrainer hyper parameters: {trainer_hyper_params}\n")

    model_name = helpers.get_module_name(model_dict)
    data_loader_name = helpers.get_module_name(data_loader_dict)

    hp_model = hp.HParam('model_class', hp.Discrete([model_name]))
    hp_dataloader = hp.HParam('dataloader_class', hp.Discrete([data_loader_name]))

    tensorboard_experiment_name = model_name + "_" + data_loader_name
    tensorboard_experiment_id = helpers.get_tensorboard_experiment_id(
        experiment_name=tensorboard_experiment_name,
        tensorboard_tracking_folder=tensorboard_tracking_folder
    )

    # Hyper parameters search for the training loop
    hp_batch_size = hp.HParam('batch_size', hp.Discrete(trainer_hyper_params["batch_size"]))
    hp_epochs = hp.HParam('epochs', hp.Discrete(trainer_hyper_params["epochs"]))
    if "dropout" in model_dict['hyper_params'].keys():
        hp_dropout = hp.HParam('dropout', hp.Discrete(model_dict['hyper_params']["dropout"]))
    else:
        # Value to indicate no dropout for the model.
        hp_dropout = hp.HParam('dropout', hp.Discrete([-1.0]))
    hp_learning_rate = hp.HParam('learning_rate', hp.Discrete(trainer_hyper_params["lr_rate"]))
    hp_patience = hp.HParam('patience', hp.Discrete(trainer_hyper_params["patience"]))

    training_dataset = training_data_loader.batch(hp_batch_size.domain.values[0])
    validation_dataset = validation_data_loader.batch(hp_batch_size.domain.values[0])

    # Main loop to iterate over all possible hyper parameters
    variation_num = 0
    for epochs in hp_epochs.domain.values:
        for learning_rate in hp_learning_rate.domain.values:
            for dropout in hp_dropout.domain.values:
                for patience in hp_patience.domain.values:
                    if dropout != -1.:
                        hparams = {
                            hp_batch_size: hp_batch_size.domain.values[0],
                            hp_model: hp_model.domain.values[0],
                            hp_dataloader: hp_dataloader.domain.values[0],
                            hp_epochs: epochs,
                            hp_dropout: dropout,
                            hp_learning_rate: learning_rate,
                            hp_patience: patience,
                        }
                    else:
                        hparams = {
                            hp_batch_size: hp_batch_size.domain.values[0],
                            hp_model: hp_model.domain.values[0],
                            hp_dataloader: hp_dataloader.domain.values[0],
                            hp_epochs: epochs,
                            hp_learning_rate: learning_rate,
                            hp_patience: patience,
                        }

                    # Copy the user config for the specific current model
                    current_user_dict = user_config_dict.copy()
                    # Add dropout
                    if dropout != -1.:
                        current_user_dict["model"]['hyper_params']["dropout"] = dropout

                    if mirrored_strategy is not None and mirrored_strategy.num_replicas_in_sync > 1:
                        with mirrored_strategy.scope():
                            model = helpers.get_online_model(user_config_dict, training_config_dict)
                    else:
                        model = helpers.get_online_model(user_config_dict, training_config_dict)

                    tensorboard_log_dir = os.path.join(tensorboard_experiment_id, str(variation_num))
                    print("Start variation id:", tensorboard_log_dir)
                    train_model(
                        model=model,
                        training_dataset=training_dataset,
                        validation_dataset=validation_dataset,
                        tensorboard_log_dir=tensorboard_log_dir,
                        hparams=hparams,
                        mirrored_strategy=mirrored_strategy,
                        epochs=epochs,
                        learning_rate=learning_rate,
                        patience=patience,
                        # Fileformat must be hdf5, otherwise bug
                        # https://github.com/tensorflow/tensorflow/issues/34127
                        checkpoints_path=os.path.join(
                            tensorboard_log_dir,
                            tensorboard_experiment_name + ".{epoch:02d}-{val_loss:.2f}.hdf5"
                        )
                    )
                    variation_num += 1

    # Save final model
    model.save(helpers.generate_model_name(user_config_dict))


def train_model(
        model,
        training_dataset,
        validation_dataset,
        tensorboard_log_dir,
        hparams,
        mirrored_strategy,
        epochs,
        learning_rate,
        patience,
        checkpoints_path
):
    """
    The training loop for a single model

    :param model: The tf.keras.Model to train
    :param training_dataset: The training dataset
    :param validation_dataset: The validation dataset to evaluate training progress
    :param tensorboard_log_dir: Path of where to store TensorFlow logs
    :param hparams: A dictionary of TensorBoard.plugins.hparams.api.hp.HParam to track on TensorBoard
    :param mirrored_strategy: A tf.distribute.MirroredStrategy on how many GPUs to use during training
    :param epochs: The epochs hyper parameter
    :param learning_rate: The learning rate hyper parameter
    :param patience: The early stopping patience hyper parameter
    :param checkpoints_path: Path of where to store TensorFlow checkpoints
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
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_weights_only=False),
    ]

    compiled_model.fit(
        training_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_dataset
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--training_cfg_path', type=str,
                        help='path to the JSON config file used to store training set parameters')
    parser.add_argument('-v', '--validation_cfg_path', type=str,
                        help='path to the JSON config file used to store validation set parameters')
    parser.add_argument('-u', '--user_cfg_path', type=str, default=None,
                        help='path to the JSON config file used to store user model, dataloader and trainer parameters')
    parser.add_argument('-t', '--tensorboard_tracking_folder', type=str, default=None,
                        help='path where to store TensorBoard data and save trained model')
    args = parser.parse_args()
    main(
        training_config_path=args.training_cfg_path,
        validation_config_path=args.validation_cfg_path,
        user_config_path=args.user_cfg_path,
        tensorboard_tracking_folder=args.tensorboard_tracking_folder
    )
