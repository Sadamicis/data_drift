from inspect import _void
import numpy as np
import pandas as pd


def rate_of_NaN(data_stream):
    #data_stream == data frame in input
    for col in data_stream.columns : 
        if col != "Id":
            n = data_stream[col].size
            n_miss = np.count_nonzero(pd.isnull(data_stream[col]))
            perc_miss = n_miss*100/n
            print_value(perc_miss,col) 
    return _void
        
def print_value(x,y):
    return(print("the rate of",y,"is", x))

        