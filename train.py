# Summary:
#   Trains the predictor

import tensorflow as tf
import panda as pd

def main():
    # TODO: Implement

    # Read data
    data_path = Path(r"/project/cq-training-1/project1/data/")
    catalog = pd.read_pickle(data_path/"catalog.helios.public.20100101-20160101.pkl")

    # Clean data: remove night values and take care of Nan values
    cleanData(catalog)

    tf.debugging.set_log_device_placement(True)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Running a matrix multiplication using two tensors...")

    # Place some tensors on the GPU
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    c = tf.matmul(a, b)
    print(c)


def cleanData(dataframe):
    newDataSet = removeNightValues(dataframe)
    newDataSet = removeNanValues(newDataSet)
    newDataSet = fillGHIValues(newDataSet)


def removeNightValues(dataframe):
    return dataframe[(dataframe.DRA_DAYTIME==1) |
                            (dataframe.TBL_DAYTIME==1) |
                            (dataframe.BND_DAYTIME==1) |
                            (dataframe.FPK_DAYTIME==1) | 
                            (dataframe.GWN_DAYTIME==1) | 
                            (dataframe.PSU_DAYTIME==1) |
                            (dataframe.SXF_DAYTIME==1)]

def removeNcdfNanValues(dataframe):
    return dataframe[dataframe['ncdf_path']!='nan']

# WIP
def fillGHIValues(dataframe):
    count = 0
    stations = ['BND', 'TBL', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF']

    #for i, row in noNcdfNan.iterrows():
    for i in range(len(dataframe)):
        for station in stations:
            #currentRow = noNcdfNan.iloc[k]
            #stationColumn = noNcdfNan
            
            #if pd.notnull(currentRow[f'{station}_GHI']) == False:
        
                # Get the previous ghi value
                #previousGHI = noNcdfNan.iloc[k-1, f'{station}_GHI']
                # Search the first nonnull ghi value
            #   j = 0
            #   while(not pd.notnull(row.shift(j)[f'{station}_GHI'])):
            #      j+=1
                
            #  nextGHI = row.shift(j)[f'{station}_GHI']
            
                # Fill the value with the average of the two
            #   print(row.shift(j)[f'{station}_GHI'])
                #noNcdfNan.at[j,f"{station}_GHI"] = (previousGHI+nextGHI)/2

             

if __name__ == '__main__':
    main()
