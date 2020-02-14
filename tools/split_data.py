import sys
import pickle
import json
import random
from datetime import timedelta, date, datetime
import argparse

import numpy as np

# Script for generating config files for training/validation/test split
# Usage: python split_data.py [-c path_to_catalog.pkl] [-l cfg_name_{0}.json]

default_catalog_path = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl'

cfg_template = {
  "stations": {
    "BND": [40.05192, -88.37309, 230],
    "TBL": [40.12498, -105.23680, 1689],
    "DRA": [36.62373, -116.01947, 1007],
    "FPK": [48.30783, -105.10170, 634],
    "GWN": [34.25470, -89.87290, 98],
    "PSU": [40.72012, -77.93085, 376],
    "SXF": [43.73403, -96.62328, 473]
  },
  "target_time_offsets": [
    "P0DT0H0M0S",
    "P0DT1H0M0S",
    "P0DT3H0M0S",
    "P0DT6H0M0S"
  ],
}


def date_range(start_date, end_date):
    for day_i in range(int((end_date - start_date).days)):
        yield start_date + timedelta(day_i)


def write_cfg_file(json_file, params):
    """Write parameters to config json file.

    :param json_file: path to output json file with {0} placeholder for split.
    :param params: list (train, val, test) of dictionary of parameters.

    """

    for (param, split) in zip(params, ['train', 'validation', 'test']):
        if param is not None:
            with open(json_file.format(split), 'w') as cfg_file_handler:
                cfg_file_handler.write(json.dumps(param, indent=2))
                cfg_file_handler.write('\n')


def lightweight_year_split(
        years_splits=(2010, 2014, 2015, 2016), seed=987):
    random.seed(seed)
    resulting_splits = []

    for i in range(len(years_splits)-1):
        year_start = date(years_splits[i], 1, 1)
        year_end = date(years_splits[i+1], 1, 1)
        current_samples = []
        for current_day in date_range(year_start, year_end):
            current_samples.append(
                datetime(
                    year=current_day.year,
                    month=current_day.month,
                    day=current_day.day,
                    hour=random.randint(0, 23),
                    minute=random.choice([0, 15, 30, 45])
                ).strftime('%Y-%m-%dT%H:%M:%S')
            )
        resulting_splits.append(current_samples)

    return np.array(resulting_splits)


def hourly_split(
        years_splits=(2010, 2014, 2015, 2016), seed=987):
    random.seed(seed)
    resulting_splits = []

    for i in range(len(years_splits)-1):
        year_start = date(years_splits[i], 1, 1)
        year_end = date(years_splits[i+1], 1, 1)
        current_samples = []
        for current_day in date_range(year_start, year_end):
            for new_hour in range(0, 24):
                current_samples.append(
                    datetime(
                        year=current_day.year,
                        month=current_day.month,
                        day=current_day.day,
                        hour=new_hour,
                        minute=random.choice([0, 15, 30, 45])
                    ).strftime('%Y-%m-%dT%H:%M:%S')
                )
        resulting_splits.append(current_samples)

    return np.array(resulting_splits)


def generate_params(catalog_file=default_catalog_path,
                    method='lightweight_year_split', **kwargs):
    """Generate parameters for config file based on a given method

    :param catalog_file: path to pickle catalog file.
    :param method: str, name of split method to use.
    :param kwargs: additional keyword arguments passed to the split method.
    :return: tuple (train, val, test) of dictionary of config parameters.

    """

    with open(catalog_file, 'rb') as df_file_handler:
        df = pickle.load(df_file_handler)
    split_method = getattr(sys.modules[__name__], method)
    target_datetimes_split = split_method(**kwargs)
    params = []
    for i in range(3):
        if target_datetimes_split[i] is None:
            params.append(None)
        else:
            params.append(
                {'dataframe_path': catalog_file,
                 'start_bound': df.index[0].date().strftime('%Y-%m-%d'),
                 'end_bound': df.index[-1].date().strftime('%Y-%m-%d'),
                 'stations': cfg_template['stations'],  # hard coded
                 'target_time_offsets': cfg_template['target_time_offsets'],  # hard coded
                 'target_datetimes': target_datetimes_split[i]})
    return tuple(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--catalog_path', type=str,
                        default=default_catalog_path,
                        help='path to the pandas catalog file')
    parser.add_argument('-l', '--label', type=str,
                        default='project1_cfg_{0}.json',
                        help='label for the parameters file, with a {0} placeholder')
    parser.add_argument('-m', '--method', type=str,
                        default='lightweight_year_split',
                        help='method to use')
    args = parser.parse_args()
    gparams = generate_params(args.catalog_path, method=args.method)
    write_cfg_file(args.label, gparams)
