

import pandas as pd
import seaborn as sns
import pymc3 as pm
import numpy as np
import theano.tensor as tt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pmlb import fetch_data

def predict(A, B, X):
    Y_pred = A.mean(axis=0) + np.dot(X, B.mean(axis=0))
    p = np.exp(Y_pred).T/np.sum(np.exp(Y_pred), axis=1)
    p_class = np.argmax(p, axis=0)
    return p, p_class

iris = sns.load_dataset("iris")
y_2 = pd.Categorical(iris['species']).labels
x_n = iris.columns[:-1]
x_2 = iris[x_n].values

x_2, tx_2, y_2, ty_2 = train_test_split(x_2, y_2, random_state=32)

with pm.Model() as modelo_s:
    alfa = pm.Normal('alfa', mu=0, sd=10, shape=3)
    beta = pm.Normal('beta', mu=0, sd=10, shape=(4,3))

    mu = alfa + pm.math.dot(x_2, beta)
    p = pm.Deterministic('p', tt.nnet.softmax(mu))

    yl = pm.Categorical('yl', p=p, observed=y_2)
    step = pm.Metropolis()
    trace_s = pm.sample(10, step)

p, p_class = predict(trace_s["alfa"], trace_s["beta"], tx_2)

print(accuracy_score(ty_2, p_class))

import xgboost as xgb

clf = xgb.XGBClassifier()
clf.fit(x_2, y_2)
p_class_ = clf.predict(tx_2)

print(accuracy_score(ty_2, p_class_))







