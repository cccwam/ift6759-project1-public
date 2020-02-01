from datetime import datetime
from pathlib import Path


def tensorboard_experiment_id(initial, experiment_name,
                              path=Path('/project/cq-training-1/project1/teams/team03/tensorboard')):
    """
        Create a unique id for tensorboard for the experiment
    :param initial: initial of user
    :param experiment_name: name of experiment
    :param path:
    :return:
    """
    return path / initial / (initial + "-" + experiment_name + "-" + datetime.utcnow().isoformat())
