from turtle import end_fill
import matplotlib.pyplot as plt
import psi
import random
from river.drift import KSWIN
import numpy as np
import pandas as pd
from scipy import stats

#testimonianza che psi non fa bene

iris=pd.read_csv("Iris.csv")
setosa = iris[iris["Species"] == "Iris-setosa"]
setosa=setosa.iloc[:,1]

a=(setosa.max()-setosa.min()) /2
print("setosa mean:" , setosa.mean())
plt.plot(range(50),setosa,c="red")

setosa_ran=np.zeros(50)
for i in range(50):
    x = -a + a*random.random()
    setosa_ran[i]=setosa[i] + x
print("setosa_ran mean:" , setosa.mean())
plt.plot(range(50),setosa_ran,c="green")

p = psi.calculate_psi(setosa, setosa_ran , buckettype='bins', buckets=10, axis=0)
print("PSI value", p)
plt.title("Setosa PL vs Setosa random")
plt.show()

test = stats.ks_2samp(setosa, setosa_ran)
print("KS value",test[1])

setosa=setosa.tolist()
setosa_ran=setosa_ran.tolist()

data= setosa + setosa_ran
kswin = KSWIN(alpha= 0.01, window_size=30, stat_size=10 )
for i, val in enumerate(data):
     in_drift, in_warning = kswin.update(val)
     if in_drift:
         print(f"Change detected at index {i}, input value: {val}")
         plt.axvline(i, c = "red")
plt.plot(range(100),data)
plt.title("drift of skwin")
plt.show()