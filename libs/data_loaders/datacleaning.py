

def removeNightValues(dataframe):

    return dataframe[(dataframe.DRA_DAYTIME == 1) |
                     (dataframe.TBL_DAYTIME == 1) |
                     (dataframe.BND_DAYTIME == 1) |
                     (dataframe.FPK_DAYTIME == 1) |
                     (dataframe.GWN_DAYTIME == 1) |
                     (dataframe.PSU_DAYTIME == 1) |
                     (dataframe.SXF_DAYTIME == 1)]


def removeNullPath(dataframe):
    return dataframe[dataframe['ncdf_path'] != 'nan']


# Since the first GHI values of the DRA station are NaN, it cannot
# inteprolate values, we will have to decide how to take of them
def fillGHI(dataframe):

    stations = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']
    for station in stations:
        dataframe[f'{station}_GHI'] = (dataframe[f"{station}_GHI"]).interpolate(method='linear')

    return dataframe
