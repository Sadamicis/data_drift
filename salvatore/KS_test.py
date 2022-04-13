
import stat
import matplotlib.pyplot as plt
import pandas as pd
import numpy

iris=pd.read_csv("Iris.csv")
setosa = iris[iris["Species"] == "Iris-setosa"]
virginica = iris[iris["Species"] == "Iris-virginica"]
from river.drift import KSWIN, DDM, ADWIN
kswin =KSWIN()
adwin=ADWIN()
kswin = KSWIN(alpha= 0.01, window_size=50, stat_size=20,window=setosa["SepalLengthCm"])

# Update drift detector and verify if change is detected
for i, val in enumerate(iris["SepalLengthCm"]):
     in_drift, in_warning = kswin.update(val)
     if in_drift:
         print(f"Change detected at index {i}, input value: {val}")
         plt.axvline(i, c= "red")
plt.plot(range(150),iris["SepalLengthCm"])
plt.title("Drift SepalLengthCm ")
plt.show()