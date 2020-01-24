import os
import sys
import pickle
import json
import random
import datetime

# Script for generating config files for training/validation/test split
# Usage: python split_data.py [path_to_catalog.pkl] [cfg_name_{0}.json]

default_catalog = '/project/cq-training-1/project1/data/catalog.helios.public.20100101-20160101.pkl'

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


def write_cfg_file(json_file, params):
    """Write parameters to config json file.

    :param json_file: path to output json file with {0} placeholder for split.
    :param params: list (train, val, test) of dictionary of parameters.

    """

    for (param, split) in zip(params, ['train', 'validation', 'test']):
        if param is not None:
            with open(json_file.format(split), 'w') as f:
                f.write(json.dumps(param, indent=2))
                f.write('\n')


def lightweight_year_split(
        df, train_year_ini=2010, train_year_fin=2013,
        val_year_ini=2014, val_year_fin=2014, test_year_ini=2015,
        test_year_fin=2015, rseed=987):
    """Lightweight train / val / test split based on years.

    :param df: pandas dataframe of catalog
    :param train_year_ini: int, initial training year
    :param train_year_fin: int, final training year
    :param val_year_ini: int, initial validation year
    :param val_year_fin: int, final validation year
    :param test_year_ini: int, initial test year
    :param test_year_fin: int, final test year
    :param rseed: int, random seed
    :return: tuple (train, val, test) of list of target datetimes

    This will extract exactly one data point from every day at a random
    hour and minute (00, 15, 30 or 45).

    """

    random.seed(rseed)
    time_split = ([], [], [])
    for i in range(3):
        if i == 0:
            year_ini = train_year_ini
            year_fin = train_year_fin
        elif i == 1:
            year_ini = val_year_ini
            year_fin = val_year_fin
        else:
            year_ini = test_year_ini
            year_fin = test_year_fin
        for year in range(year_ini, year_fin + 1):
            used_days = []
            for ts in df.index:
                if ts.year == year and (ts.month, ts.day) not in used_days:
                    hour = random.randint(0, 23)
                    minute = random.choice([0, 15, 30, 45])
                    dt = datetime.datetime(ts.year, ts.month, ts.day, hour, minute)
                    time_split[i].append(dt.strftime('%Y-%m-%dT%H:%M:%S'))
                    used_days.append((ts.month, ts.day))
    return time_split


def generate_params(catalog_file=default_catalog, method='lightweight_year_split', **args):
    """Generate parameters for config file based on a given method

    :param catalog_file: path to pickle catalog file.
    :param method: str, name of split method to use.
    :param args: additional keyword arguments passed to the split method.
    :return: tuple (train, val, test) of dictionary of config parameters.

    """

    with open(catalog_file, 'rb') as f:
        df = pickle.load(f)
    target_datetimes_split = globals()[method](df, **args)
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
    if (len(sys.argv) > 1) and os.path.isfile(sys.argv[1]):
        catalog = sys.argv[1]
    else:
        catalog = default_catalog
    if len(sys.argv) > 2:
        cfg_name = sys.argv[2]
    else:
        cfg_name = 'project1_cfg_{0}.json'
    gparams = generate_params(catalog)
    write_cfg_file(cfg_name, gparams)
