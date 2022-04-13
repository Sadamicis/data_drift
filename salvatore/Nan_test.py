from none import rate_of_NaN
import pandas as pd
import numpy as np
df=pd.read_csv("miss_iris.csv")
print(df.head())
rate_of_NaN(df) 