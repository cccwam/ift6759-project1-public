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

def removeNanValues(dataframe):
    pass


def fillGHIValues(dataframe):
    pass

if __name__ == '__main__':
    main()
