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
    --user_cfg_path configs/user/cnn_image_v4_for_evaluator.json \
    --stats_output_path stat_output.txt
```

* `args[0]`: Path where the raw model predictions should be saved (for visualization purposes).
* `args[1]` Path to the JSON config file used to store test set/evaluation parameters.
* `user_cfg_path` Path to the JSON config file used to store user model/dataloader parameters.
* `stats_output_path` Path where the prediction stats should be saved (for benchmarking).

## Trainer.py example usage

```
python trainer.py \
    --admin_cfg_path configs/admin/daily_daytime_01_train.json \
    --validation_cfg_path configs/admin/daily_daytime_01_validation.json \
    --user_cfg_path configs/user/ineichen_clear_sky_v1.json \
    --tensorboard_tracking_folder /project/cq-training-1/project1/teams/team03/tensorboard/$USER
```

* `admin_cfg_path`: Path to the JSON config file used to store train set parameters.
* `user_cfg_path`: Path to the JSON config file used to store user model/dataloader parameters.
* `tensorboard_tracking_folder`: Path where to store tensorboard data and save trained model. 

