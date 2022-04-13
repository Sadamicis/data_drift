import stat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
fun1 = lambda x : np.abs(np.sin(x))
fun2 = lambda x : np.sin(x)
n=2000
pred = np.zeros(n)
y = np.zeros(n)
x = np.zeros(n)
for i in range(n):
  x[i] = 100*random.random()
x.sort()
y=fun1(x)
y[500:800]=fun2(x[500:800])
y[1000:1300]=fun2(x[1000:1300])
y[1500:1800]=fun2(x[1500:1800])
pred = y

from river.drift import PageHinkley, EDDM, ADWIN, KSWIN
drift_det=ADWIN()
for i, val in enumerate(pred):
   in_drift, in_warning = drift_det.update(val)
   if in_warning:
       plt.axvline(i,color="green")
   if in_drift:
      print(f"Change detected at index {i}, input value: {val}")
      plt.axvline(i,color="red")
plt.plot(range(2000),pred)
plt.show()