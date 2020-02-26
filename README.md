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
