

# from http://docs.pymc.io/notebooks/GLM-logistic.html

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn
import warnings

from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
from collections import OrderedDict
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import fmin_powell
from scipy import integrate

import theano as thno
import theano.tensor as T


def run_models(df, upper_order=5):
    '''
    Convenience function:
    Fit a range of pymc3 models of increasing polynomial complexity.
    Suggest limit to max order 5 since calculation time is exponential.
    '''

    models, traces = OrderedDict(), OrderedDict()

    for k in range(1,upper_order+1):

        nm = 'k{}'.format(k)
        fml = create_poly_modelspec(k)

        with pm.Model() as models[nm]:

            print('\nRunning: {}'.format(nm))
            pm.glm.GLM.from_formula(fml, df, family=pm.glm.families.Normal())

            traces[nm] = pm.sample(2000, init=None)

    return models, traces

def plot_traces(traces, retain=1000):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:], figsize=(12,len(traces.varnames)*1.5),
        lines={k: v['mean'] for k, v in pm.df_summary(traces[-retain:]).iterrows()})

    for i, mn in enumerate(pm.df_summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')

def create_poly_modelspec(k=1):
    '''
    Convenience function:
    Create a polynomial modelspec string for patsy
    '''
    return ('income ~ educ + hours + age ' + ' '.join(['+ np.power(age,{})'.format(j) for j in range(2,k+1)])).strip()



data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, names=['age', 'workclass', 'fnlwgt',
                'education-categorical', 'educ',
                'marital-status', 'occupation',
                'relationship', 'race', 'sex',
                'captial-gain', 'capital-loss',
                'hours', 'native-country',
                'income'])


income = 1 * (data['income'] == " >50K")
age2 = np.square(data['age'])

data = data[['age', 'educ', 'hours']]
data['age2'] = age2
data['income'] = income


with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula('income ~ age + age2 + educ + hours', data, family=pm.glm.families.Binomial())
    trace_logistic_model = pm.sample(500)


sd = np.mean(trace_logistic_model['sd'])
intercept = np.mean(trace_logistic_model['Intercept'])
age = np.mean(trace_logistic_model['age'])
age2 = np.mean(trace_logistic_model['age2'])
edu = np.mean(trace_logistic_model['education'])
hours = np.mean(trace_logistic_model['hours'])

eval_set = data
Y = eval_set['income'].tolist()
yp = eval_set['age'].values * age + eval_set['age2'].values * age2 \
    + eval_set['education'].values * edu + eval_set['hours'].values \
    * hours + intercept


th = np.mean(yp)
yp_b = [1 if p >= th else 0 for p in yp]

print(accuracy_score(Y, yp_b))

plt.plot(Y[:80], color="green")
plt.plot(yp[:80], color="pink")
plt.show()
