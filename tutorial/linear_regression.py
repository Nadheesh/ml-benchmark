
#from http://docs.pymc.io/notebooks/getting_started.html#A-Motivating-Example:-Linear-Regression


import numpy as np
import matplotlib.pyplot as plt

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 200

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
FY = alpha + beta[0]*X1 + beta[1]*X2

X1, TX1 = X1[100:], X1[:100]
X2, TX2 = X2[100:], X2[:100]
Y, TY = FY[100:], FY[:100]



# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,4))
# axes[0].scatter(X1, Y)
# axes[1].scatter(X2, Y)
# axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2')
# plt.show()


import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

map_estimate = pm.find_MAP(model=basic_model)
# map_estimate = pm.find_MAP(model=basic_model, fmin=optimize.fmin_powell)
print(map_estimate)

alpha1 = map_estimate['alpha']
beta1 = map_estimate['beta']
sigma1 = map_estimate['sigma']
yp = (alpha1 + beta1[0]*TX1 + beta1[1]*TX2)
# print(TY)
# print(yp)
plt.plot(FY, color="green")
plt.plot(yp, color="pink")
plt.show()