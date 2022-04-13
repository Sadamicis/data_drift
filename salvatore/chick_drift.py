from matplotlib import pyplot as plt
from river import datasets
import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, linear_model
from sklearn import metrics
#import shap

p=datasets.ChickWeights()
data=pd.read_csv(p.path)
plt.scatter(data.time,data.weight,c=data.diet) 
plt.title("Crescita peso pulcini in 21 giorni")
plt.colorbar()  
plt.show()
chick1=data[(data["diet"] == 2) | (data["diet"] == 1)]
chick3 = data[data["diet"]==3]
c_train = chick1[(chick1["chick"]!=5) & (chick1["chick"]!=10) &
                  (chick1["chick"]!=15) & (chick1["chick"]!=20) &
                  (chick1["chick"]!=25) & (chick1["chick"]!=30)]
c_test = chick1[(chick1["chick"]==5) | (chick1["chick"]==10) |
                  (chick1["chick"]==15) | (chick1["chick"]==20) |
                  (chick1["chick"]==25) | (chick1["chick"]==30)]

reg = linear_model.LogisticRegression(max_iter=10000)
reg.fit(c_train[["time", "diet"]],c_train["weight"])
pred=reg.predict(c_test[["time","diet"]])
print("MAPE:",metrics.mean_absolute_percentage_error(c_test["weight"],pred))

plt.scatter(c_test.time,pred,c="green")
plt.scatter(c_test.time,c_test.weight,c="red")
plt.title("predizioni vs test")
plt.show()

from scipy import stats
p_value = 0.05
rejected = 0
import psi
col="weight"
test = stats.ks_2samp(chick1[col], chick3[col])
p = psi.calculate_psi(chick3[col], chick1[col] , buckettype='bins', buckets=10, axis=0)
print("psi value", p, "of", col)
if test[1] < p_value:
    rejected += 1
    print("Column rejected", col)
print("We rejected",rejected,"columns in total")

pred=reg.predict(chick3[["time", "diet"]])
print("MAPE:", metrics.mean_absolute_percentage_error(chick3.weight,pred))

plt.scatter(chick3.time,pred,c="green")
plt.scatter(chick3.time,chick3.weight,c="red")
plt.title("predizioni vs test, diet 3")
plt.show()
"""
explainer = shap.explainers.Exact(reg.predict_proba,c_train[["time", "diet"]] )
shap_values = explainer(c_train[["time", "diet"]])

# get just the explanations for the positive class
shap_values = shap_values[...,1]
shap.plots.bar(shap_values)

"""