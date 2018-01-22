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

import warnings

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import xgboost
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    root_folder = "dataset"

    if _use_test_file:
        train_df = pd.read_csv("{0}/{1}/train.csv".format(root_folder, _name))
        test_df = pd.read_csv("{0}/{1}/test.csv".format(root_folder, _name))
        split_index = train_df.shape[0]
        df = train_df.append(test_df)
    else:
        df = pd.read_csv("{0}/{1}/{1}.csv".format(root_folder, _name))

    obj_columns = df.select_dtypes(['object']).columns
    for col in obj_columns:
        df[col] = df[col].astype("str").astype("category")

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

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
        "glass": ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'],
        "titanic": ['pclass', 'sex', 'age', 'survived'],  # 'fare', 'parch'
        "iris": ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
        "adult": ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation',
                  'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                  'target'],
        "optdigits": ['target']
    }

    return def_load_dataset(name,
                            features.get(name)[:-1], features.get(name)[-1], params["use_test_file"],
                            params["feature_selection"])


def feature_selection(data: pd.DataFrame, label_col, num_features=6, criteria="corr"):
    corr_score = (data.corr().abs())[label_col].sort_values(ascending=False)
    return corr_score.head(num_features + 1).index.tolist()[1:]


def train_bayesian_model(X, y, _use_map):
    class_count = len(set(y.tolist()))
    with pm.Model() as modelo_s:
        alfa = pm.Normal('alfa', mu=1, sd=10, shape=class_count)
        beta = pm.Normal('beta', mu=1, sd=10, shape=(X.shape[1], class_count))

        mu = alfa + pm.math.dot(X, beta)
        p = pm.Deterministic('p', tt.nnet.softmax(mu))

        yl = pm.Categorical('yl', p=p, observed=y)
        step = pm.Metropolis()
        if _use_map:
            map_trace = pm.find_MAP()
            trace_s = pm.sample(500, start=map_trace, step=step)

        else:
            trace_s = pm.sample(500, step=step)

    if _use_map:
        return trace_s, map_trace
    return trace_s


def predict_bayesian_model(alfa, beta, X):
    Y_pred = alfa.mean(axis=0) + np.dot(X, beta.mean(axis=0))
    p = np.exp(Y_pred).T / np.sum(np.exp(Y_pred), axis=1)
    p_class = np.argmax(p, axis=0)
    return p_class


def predict_bayesian_map_model(alfa, beta, X):
    Y_pred = alfa + np.dot(X, beta)
    p = np.exp(Y_pred).T / np.sum(np.exp(Y_pred), axis=1)
    p_class = np.argmax(p, axis=0)
    return p_class


def eval_bayesian_models(_train_X, _test_X, _train_y, _test_y):
    trace = train_bayesian_model(_train_X, _train_y, False)
    prediction = predict_bayesian_model(trace["alfa"], trace["beta"], _test_X)

    bayesian_accr = accuracy_score(_test_y, prediction)
    print("Bayesian Accuracy without MAP start= {0}".format(bayesian_accr))

    trace, map_trace = train_bayesian_model(_train_X, _train_y, True)

    prediction = predict_bayesian_model(trace["alfa"], trace["beta"], _test_X)
    bayesian_map_start_accr = accuracy_score(_test_y, prediction)
    print("Bayesian Accuracy with MAP start = {0}".format(bayesian_map_start_accr))

    map_prediction = predict_bayesian_map_model(map_trace["alfa"], map_trace["beta"], _test_X)
    bayesian_map_accr = accuracy_score(_test_y, map_prediction)
    print("Bayesian(MAP) Accuracy = {0}".format(bayesian_map_accr))

    return bayesian_accr, bayesian_map_start_accr, bayesian_map_accr


def eval_baseline_models(_train_X, _test_X, _train_y, _test_y):
    xgb = xgboost.XGBClassifier()
    xgb.fit(_train_X, _train_y)
    _pred = xgb.predict(_test_X)
    xgb_accr = accuracy_score(_test_y, _pred)
    print("XGBoost accuracy = {0}".format(xgb_accr))

    rfc = RandomForestClassifier()
    rfc.fit(_train_X, _train_y)
    _pred = rfc.predict(_test_X)
    rfc_accr = accuracy_score(_test_y, _pred)
    print("Random Forest accuracy = {0}".format(rfc_accr))

    return xgb_accr, rfc_accr


# def eval_model(train, test, features,):

# dataset_name : [dataset_name, features to select, use_train_test_files]
datasets = [{"name": "optdigits", "use_test_file": True},
            {"name": "optdigits", "feature_selection": 30, "use_test_file": True},
            {"name": "adult", "feature_selection": 6, "use_test_file": True},
            {"name": "adult", "use_test_file": True},
            {"name": "glass", "feature_selection": 4, "use_test_file": False},
            {"name": "glass", "use_test_file": False},
            {"name": "iris", "use_test_file": False},
            {"name": "titanic", "use_test_file": False},
            ]  # classification_dataset_names

label = "target"

results = pd.DataFrame(
    columns=["Dataset", "Features", "XGBoost", "Random Forest Classifier", "Bayesian(without MAP start)",
             "Bayesian(with MAP start)",
             "Bayesian(MAP)"])
for classification_dataset in datasets:
    print("Dataset : {0}".format(classification_dataset))
    train_X, test_X, train_y, test_y = data_loader(**classification_dataset)

    # features = feature_selection(train, label)
    features = train_X.columns.tolist()
    print(features)

    xgb_accr, rfc_accr = eval_baseline_models(train_X, test_X, train_y, test_y)
    bayesian_accr, bayesian_map_start_accr, bayesian_map_accr = eval_bayesian_models(train_X, test_X, train_y, test_y)

    results.loc[results.shape[0]] = [classification_dataset["name"],
                                     features,
                                     xgb_accr,
                                     rfc_accr,
                                     bayesian_accr,
                                     bayesian_map_start_accr,
                                     bayesian_map_accr]

results.to_csv("classification_evaluation.csv")
