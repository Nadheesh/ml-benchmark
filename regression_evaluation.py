"""
Copyright (c) , WSO2 Inc. (http://www.wso2.org) All Rights Reserved.

WSO2 Inc. licenses this file to you under the Apache License,
Version 2.0 (the "License"); you may not use this file except
in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
import re
import warnings

import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.base import TransformerMixin
from sklearn.datasets import load_diabetes, load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def def_load_dataset(_name, _features, _label, _use_test_file, _feature_selection):
    root_folder = "dataset/regression"

    if _use_test_file:
        train_df = pd.read_csv("{0}/{1}/train.csv".format(root_folder, _name))
        test_df = pd.read_csv("{0}/{1}/test.csv".format(root_folder, _name))
        split_index = train_df.shape[0]
        df = train_df.append(test_df)
    elif _name == "diabetes":
        _data = load_diabetes()
        df = pd.DataFrame(data=np.c_[_data['data'], _data['target']],
                          columns=["Attr{0}".format(i) for i in range(_data['data'].shape[1])] + ['target'])
    elif _name == "boston":
        _data = load_boston()
        df = pd.DataFrame(data=np.c_[_data['data'], _data['target']],
                          columns=_data['feature_names'].tolist() + ['target'])
    else:
        df = pd.read_csv("{0}/{1}/{1}.csv".format(root_folder, _name))

    obj_columns = df.select_dtypes(['object']).columns
    for col in obj_columns:
        df[col] = df[col].astype("str").astype("category")

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # column_names are made compatible with GLM (pymc3)
    df.columns = [re.sub(" ", "_", col) for col in df.columns]

    if _feature_selection > 0:
        _features = feature_selection(df, _label, num_features=_feature_selection)

    if len(_features) <= 0:
        _features = df.columns.tolist()
        _features.remove(_label)

    X = df[_features]

    # X = X.fillna(df.median)
    X = DataFrameImputer().fit_transform(X)
    y = df[_label]

    if _use_test_file:
        return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
    return train_test_split(X, y)


def data_loader(**kwargs):
    params = {
        "name": "adult",
        "feature_selection": 0,
        "use_test_file": False
    }
    params.update(kwargs)

    name = params["name"]

    features = {
        "boston": ['target'],
        "diabetes": ['target'],  # 'fare', 'parch'
        "finance_distress": ['Financial_Distress'],
        "kc_house_data": ['price'],
        "winequality_red": ['quality']
    }

    return def_load_dataset(name,
                            features.get(name)[:-1], features.get(name)[-1], params["use_test_file"],
                            params["feature_selection"])


def feature_selection(data: pd.DataFrame, label_col, num_features=6, criteria="corr"):
    corr_score = (data.corr().abs())[label_col].sort_values(ascending=False)
    return corr_score.head(num_features + 1).index.tolist()[1:]


def train_bayesian_model(X, y, _use_map):
    # with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    #     # Define priors
    #     sigma = pm.HalfCauchy('sigma', beta=10, testval=1., shape=1)
    #     intercept = pm.Normal('Intercept', 0, sd=20, shape=1)
    #     x_coeff = pm.Normal('x', 0, sd=20, shape=(X.shape[1], 1))
    #
    #     mu = intercept + pm.math.dot(X, x_coeff)
    #     # Define likelihood
    #     likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y)
    #
    #     # Inference!
    #     trace = pm.sample(1)

    features = X.columns.tolist()
    _data = pd.DataFrame(data=X.values, columns=X.columns, index=X.index)
    _data["y"] = y

    try:
        with pm.Model() as model:
            pm.glm.GLM.from_formula('y ~ {0}'.format("+".join(features)), _data)

            if _use_map:
                map_trace = pm.find_MAP()
                trace = pm.sample(500, start=map_trace)
            else:
                trace = pm.sample(500)

        if _use_map:
            return trace, map_trace
        return trace

    except ValueError:
        raise Exception()


def predict_bayesian_model(trace, X):
    _features = X.columns

    prediction = np.zeros(X.shape[0])
    for feature in _features:
        prediction += X[feature] * np.mean(trace[feature])
    prediction += np.mean(trace["Intercept"])

    # Y_pred = intercept.mean(axis=0) + np.dot(X, x_coff.mean(axis=0))
    # p = np.exp(Y_pred).T / np.sum(np.exp(Y_pred), axis=1)
    # p_class = np.argmax(p, axis=0)
    return prediction


def eval_bayesian_models(_train_X, _test_X, _train_y, _test_y):
    bayesian_err = bayesian_map_err = bayesian_map_start_err = -1

    try:
        trace = train_bayesian_model(_train_X, _train_y, False)
        prediction = predict_bayesian_model(trace, _test_X)

        bayesian_err = mean_squared_error(_test_y, prediction)
        print("Bayesian Error without MAP start= {0}".format(bayesian_err))

        trace, map_trace = train_bayesian_model(_train_X, _train_y, True)
        prediction = predict_bayesian_model(trace, _test_X)
        bayesian_map_start_err = mean_squared_error(_test_y, prediction)
        print("Bayesian Error with MAP start = {0}".format(bayesian_map_start_err))

        prediction = predict_bayesian_model(map_trace, _test_X)
        bayesian_map_err = mean_squared_error(_test_y, prediction)
        print("Bayesian(MAP) Error = {0}".format(bayesian_map_err))

    except Exception:
        print("Error when training the models")
    return bayesian_err, bayesian_map_start_err, bayesian_map_err


def eval_baseline_models(_train_X, _test_X, _train_y, _test_y):
    lasso = Lasso(normalize=True)
    lasso.fit(_train_X, _train_y)
    _pred = lasso.predict(_test_X)
    lasso_err = mean_squared_error(_test_y, _pred)
    print("Lasso regression MSE Error = {0}".format(lasso_err))

    rfr = RandomForestRegressor(n_estimators=20)
    rfr.fit(_train_X, _train_y)
    _pred = rfr.predict(_test_X)
    rfr_err = mean_squared_error(_test_y, _pred)
    print("Random Forest MSE Error = {0}".format(rfr_err))

    return lasso_err, rfr_err


# def eval_model(train, test, features,):

# dataset_name : [dataset_name, features to select, use_train_test_files]
datasets = [
    {"name": "kc_house_data", "use_test_file": False},
    {"name": "finance_distress", "use_test_file": False},
    {"name": "kc_house_data", "use_test_file": False, "feature_selection": 6},
    {"name": "finance_distress", "use_test_file": False, "feature_selection": 6},
    {"name": "boston", "use_test_file": False},
    {"name": "winequality_red", "use_test_file": False},
    {"name": "diabetes", "use_test_file": False},
]

label = "target"

results = pd.DataFrame(
    columns=["Dataset", "Features", "Lasso Regression", "Random Forest Regressor", "Bayesian(without MAP start)",
             "Bayesian(with MAP start)",
             "Bayesian(MAP)"])
for regression_dataset in datasets:
    print("Dataset : {0}".format(regression_dataset))
    train_X, test_X, train_y, test_y = data_loader(**regression_dataset)

    # features = feature_selection(train, label)
    features = train_X.columns.tolist()
    print(features)

    lasso_err, rfr_err = eval_baseline_models(train_X, test_X, train_y, test_y)
    bayesian_err, bayesian_map_start_err, bayesian_map_err = eval_bayesian_models(train_X, test_X, train_y, test_y)

    print()
    results.loc[results.shape[0]] = [regression_dataset["name"],
                                     features,
                                     lasso_err,
                                     rfr_err,
                                     bayesian_err,
                                     bayesian_map_start_err,
                                     bayesian_map_err]

results.to_csv("regression_evaluation(500+500).csv")
