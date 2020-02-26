# ift6759-project1

Project 1 solution from Team 3 in IFT6759: Advanced Projects
at University of Montreal.

## Notable files

* trainer.py
  * Used to train the predictor.
* evaluator.py
  * Used to evaluate the predictor.
* tools/
  * Helper scripts to facilitate development.
* requirements.txt
  * All the requirements necessary to run the code.

## Guide to our configuration files

In this project, we utilise the admin configuration file format provided by the instructors. 
These can be found in configs/admin. We generated these files using our tools/split_data.py.
Additionally, we defined user configuration files. These can be found in configs/user.

The user configuration file schema can be found at configs/user/schema.json. All in all, it
contains the following configurable properties:

* model
    * hyper_params: The hyper parameters for the model (some are defined as a list to allow
    for a hyper-parameter search during training)
    * source: This parameter is used during evaluation and determines how to load the trained model. Either:
        * "" (defaults loading model during evaluation to ../model/best_model.hdf5)
        * Absolute path to the hdf5 file 
        * "online": Means that the model does not have a hdf5 file and it will be loaded without any
        predefined weights. 
* data_loader
    * hyper_params: The hyper parameters for the data loader
    * should_preprocess_data: True if pre-processing should be done right before evaluation.
    If false, looks into the property preprocessed_data_source.test for where to load prematurely pre-processed data.
    For background, we usually used our tools/netcdf_crop.py to prematurely pre-process our data and then
    re-trained on this data to speed up launching experiments.
    * preprocessed_data_source:
        * training: Location of prematurely pre-processed training NetCDF files that have been outputted by
        tools/netcdf_crop.py 
        * training: Location of prematurely pre-processed validation NetCDF files that have been outputted by
        tools/netcdf_crop.py 
        * test: Location of prematurely pre-processed test NetCDF files that have been outputted by
        tools/netcdf_crop.py.
* trainer 
    * hyper_params: We have defined that the following trainer hyper parameters are required by our trainer:
        * lr_rate: The learning rate
        * batch_size: The batch size
        * epochs: The number of epochs
        * patience: How patient we should be before we perform early stopping during training

## Sbatch job example

The the folder tools, there's a file called sbatch_template.sh. 
It is currently set up to run a training of our best model configuration. 
Simply run `sbatch sbatch_template.sh` to launch the training job.

## Evaluator.py example usage

```
python evaluator.py \
    pred_output.txt \
    configs/admin/dummy_test_cfg.json \
    --user_cfg_path eval_user_cfg.json \
    --stats_output_path stat_output.txt
```

* `args[0]`: Path where the raw model predictions should be saved (for visualization purposes).
* `args[1]` Path to the JSON config file used to store test set/evaluation parameters.
* `user_cfg_path` Path to the JSON config file used to store user model, dataloader and trainer parameters.
* `stats_output_path` Path where the prediction stats should be saved (for benchmarking).

IMPORTANT: The evaluation step first preprocess data and writes netcdf files to disk.
This is the path specified in the [data_loader -> hyper_params -> preprocessed_data_source -> test]
section of user_cfg_path. The default path provided should have read/write permission for
everyone. If multiple tests are to be run concurrently, they must point to
different test path to avoid overwriting files from another test.

## Trainer.py example usage

```
python trainer.py \
    --training_cfg_path configs/admin/daily_daytime_01_train.json \
    --validation_cfg_path configs/admin/daily_daytime_01_validation.json \
    --user_cfg_path configs/user/ineichen_clear_sky_v1.json \
    --tensorboard_tracking_folder /project/cq-training-1/project1/teams/team03/tensorboard/$USER
```

* `training_cfg_path`: Path to the JSON config file used to store training set parameters.
* `validation_cfg_path`: Path to the JSON config file used to store validation set parameters.
* `user_cfg_path`: Path to the JSON config file used to store user model, dataloader and trainer parameters.
* `tensorboard_tracking_folder`: Path where to store TensorBoard data and save trained model. 
