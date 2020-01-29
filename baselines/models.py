import datetime
import typing
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

def global_mean(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """This function should be modified in order to prepare & return your own prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A model
    """

    class DummyModel(tf.keras.Model):

      def __init__(self, target_time_offsets):
          super(DummyModel, self).__init__()
          self.train()

      def train(self):
          station_names = ['BND', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
          with open(config['dataframe_path'], 'rb') as f:
                df = pickle.load(f)
                for station_name in station_names:
                    setattr(self, station_name, df[f"{station_name}_GHI"].mean())
                self.global_mean = (self.BND + self.DRA + self.FPK + self.GWN + self.PSU + self.SXF + self.TBL) / 7.

      def call(self, inputs):
        stations_lat_id = {'BND': 40051,
                           'TBL': 40124,
                           'DRA': 36623,
                           'FPK': 48307,
                           'GWN': 34254,
                           'PSU': 40720,
                           'SXF': 43734}
        preds = np.zeros([inputs.shape[0], 4])
        for i in range(inputs.shape[0]):
            # mydate = datetime.datetime(inputs[i,0], inputs[i,1], inputs[i,2], inputs[i,3], inputs[i,4])
            if config['model_mode'] == 'per_station':
                for station, station_lat_id in stations_lat_id.items():
                    if inputs[i, 5] == station_lat_id:
                        preds[i, :] = getattr(self, station)
                        break
                else:
                    raise NotImplementedError()
            elif config['model_mode'] == 'global':
                preds[i, :] = self.global_mean
        return preds

    model = DummyModel(target_time_offsets)

    return model


def memorize_data(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """This function should be modified in order to prepare & return your own prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A model
    """

    class MemorizeData(tf.keras.Model):

      def __init__(self, target_time_offsets):
          super(MemorizeData, self).__init__()
          self.train()

      def train(self):
          station_names = ['BND', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
          with open(config['dataframe_path'], 'rb') as f:
                df = pickle.load(f)
                self.datetimes = df.index
                for station_name in station_names:
                    setattr(self, station_name, df[f"{station_name}_GHI"])

      def call(self, inputs):
        if len(list(stations.keys())) > 1:
            raise NotImplementedError()
        station_name = list(stations.keys())[0]
        preds = np.zeros([inputs.shape[0], 4])
        for i in range(inputs.shape[0]):
            mydate = datetime.datetime(inputs[i,0], inputs[i,1], inputs[i,2], inputs[i,3], inputs[i,4])
            for l in range(4):
                k = self.datetimes.get_loc(mydate + target_time_offsets[l])
                z = getattr(self, station_name)[k]
                if np.isnan(z):
                    z = 0
                preds[i, l] = z * config['rmse_test_scale_factor']
        return preds

    model = MemorizeData(target_time_offsets)

    return model


def ineichen(
        stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
):
    """This function should be modified in order to prepare & return your own prediction model.

    Args:
        stations: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
        target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.

    Returns:
        A model
    """

    class Ineichen(tf.keras.Model):

      def __init__(self, target_time_offsets):
          super(Ineichen, self).__init__()
          self.train()

      def train(self):
          pass

      def call(self, inputs):
        preds = np.zeros([inputs.shape[0], 4])
        for i in range(inputs.shape[0]):
            mydate = datetime.datetime(inputs[i,0], inputs[i,1], inputs[i,2], inputs[i,3], inputs[i,4])
            import pvlib
            latitude, longitude = inputs[i,5] / 1000., inputs[i, 6] / 1000.
            altitude = inputs[i,7]
            times = pd.date_range(start=mydate.isoformat(),
                                  end=(mydate+target_time_offsets[3]).isoformat(), freq='1H')
            solpos = pvlib.solarposition.get_solarposition(times, latitude,
                                                           longitude)
            apparent_zenith = solpos['apparent_zenith']
            airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
            pressure = pvlib.atmosphere.alt2pres(altitude)
            airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
            linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times,
                                                                    latitude,
                                                                    longitude)
            dni_extra = pvlib.irradiance.get_extra_radiation(times)
            # an input is a pandas Series, so solis is a DataFrame
            ineichen = pvlib.clearsky.ineichen(apparent_zenith, airmass,
                                               linke_turbidity, altitude,
                                               dni_extra)
            preds[i, 0] = ineichen.ghi[0]
            preds[i, 1] = ineichen.ghi[1]
            preds[i, 2] = ineichen.ghi[3]
            preds[i, 3] = ineichen.ghi[6]
        return preds

    model = Ineichen(target_time_offsets)

    return model

