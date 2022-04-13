import numpy as np
from river.drift import DDM, ADWIN
import random

import matplotlib.pyplot as plt
ddm = DDM(min_num_instances=30)
adwin=ADWIN()
fun1 = lambda x : np.abs(np.sin(x))
#fun1 = lambda x : x**2
fun2 = lambda x : np.sin(x)
N=100
pred = np.zeros(N)
y = np.zeros(N)
x = np.zeros(N)

for i in range(N):
  x[i] = 6*random.random()
  y[i] = -1 + 2 * random.random()
x.sort()
plt.plot(x, fun1(x))
plt.plot(x, fun2(x))
plt.title("model")
plt.show()
for i in range(N):
 if (y[i] > fun1(x[i])) and (y[i] < fun2(x[i])):
      pred[i] = 1
 if (y[i] < fun1(x[i])) and (y[i] > fun2(x[i])):
     pred[i] = 1
plt.scatter(x, y, c=pred)

for i, val in enumerate(pred):
   in_drift, in_warning = ddm.update(val)
   if in_drift:
     print(f"Change detected at index {in_drift} {in_warning}  {i}, input value: {val}")
     plt.scatter(x[i],y[i], color="red")
plt.title("Drift ddm")
plt.show()



