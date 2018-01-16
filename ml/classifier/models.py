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

import abc
import xgboost as xgb


class BaseClassifier(object):
    """
    Base classifier
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        self.init_model(**kwargs)

    @abc.abstractclassmethod
    def fit(self, X, y):
        """
        Creates the model from data provided
        :param X: Iterable of texts
        :param y: Respective labels of texts
        :return: list of predictions
        """
        pass

    @abc.abstractclassmethod
    def predict(self, X):
        """
        Predict categories for given list of inputs
        :param X: Iterable of texts
        :return: list of predictions
        """
        pass

    @abc.abstractclassmethod
    def init_model(self, **kwargs):
        """
        Initiate the model using kwargs
        :param kwargs:
        :return:
        """
        pass


class XGBoost(BaseClassifier):
    """
    XGBoost classifier
    """

    def init_model(self, **kwargs):
        params = {
            "max_depth": 3,
            "n_estimators": 300,
            "learning_rate": 0.05
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
