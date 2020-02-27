# ift6759-project1

Project 1 solution from Team 3 in IFT6759: Advanced Projects
at University of Montreal.

Team members:
* Blaise Gauvin St-Denis 
* François Mercier 
* Helgi Tomas Gislason
* Ilyas Amaniss

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
    * definition:
        * module: The Python module that contains the definition for the model that should be used
        * name: The Python name that is the model definition
    * hyper_params: The hyper parameters for the model (some are defined as a list to allow
    for a hyper-parameter search during training)
    * source: This parameter is used when the code is run using evaluator.py. It determines
    how to load the trained model. It's value should be one of the following:
        * "" (defaults loading model during evaluation to ../model/best_model.hdf5)
        * Absolute path to the hdf5 file 
        * "online": Means that the model does not have a .hdf5 file and it will be loaded
        without any predefined weights. 
* data_loader
    * definition:
        * module: The Python module that contains the definition for the data_loader that should be used
        * name: The Python name that is the data_loader definition
    * hyper_params: The hyper parameters for the data_loader
    * should_preprocess_data: This parameter is only supported using evaluator.py, it is not supported
    for trainer.py. If one wants to pre-process data before using trainer.py, one must use tools/netcdf_crop.py
    manually first and set preprocessed_data_source.training as the value to where the resulting pre-processed
    files are stored. Likewise, preprocessed_data_source.validation should have the pre-processed NetCDF files
    which should be used for validation to measure the RMSE of the model during training.
        * If true and evaluator.py is being used, the NetCDF files referenced by the dataframe referenced in
        the admin configuration file will be pre-processed using tools/netcdf_crop.py and the result will be
        stored in preprocessed_data_source.test (see below).
        * If false and evaluator.py is being used, the folder of preprocessed_data_source.test will be checked
        for pre-processed NetCDF files outputted by tools/netcdf_crop.py to use for validation.
    * preprocessed_data_source:
        * training: Location of prematurely pre-processed training NetCDF files that have been outputted by
        tools/netcdf_crop.py. This is where the data_loader will look for training data when using trainer.py.
        * training: Location of prematurely pre-processed validation NetCDF files that have been outputted by
        tools/netcdf_crop.py. This is where our data_loader will look for validation data when using trainer.py.
        * test: Location of prematurely pre-processed test NetCDF files that have been outputted by
        tools/netcdf_crop.py. If should_preprocess_data was set to true and if using evaluator.py, this is both the
        location where we will place pre-processed data outputted by tools/netcdf_crop.py (which uses the dataframe
        in the admin file as an input) and where the data_loader will read the data from during prediction time.
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
    configs/admin/daily_daytime_01_test.json \
    --user_cfg_path configs/user/cnn_image_daily_daytime_v2_pretrained.json \
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

Note: Read the section `Sbatch job example` above for information on how to run an sbatch job to train a model.

```
python trainer.py \
    --training_cfg_path configs/admin/daily_daytime_01_train.json \
    --validation_cfg_path configs/admin/daily_daytime_01_validation.json \
    --user_cfg_path configs/user/cnn_image_daily_daytime_v2_pretrained.json \
    --tensorboard_tracking_folder /project/cq-training-1/project1/teams/team03/tensorboard/$USER
```

* `training_cfg_path`: Path to the JSON config file used to store training set parameters.
* `validation_cfg_path`: Path to the JSON config file used to store validation set parameters.
* `user_cfg_path`: Path to the JSON config file used to store user model, dataloader and trainer parameters.
* `tensorboard_tracking_folder`: Path where to store TensorBoard data and save trained model. 
