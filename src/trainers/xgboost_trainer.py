import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


class Trainer:

    def __init__(self, **params):
        self.params = params
        self.model = xgb.XGBClassifier(**params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test, y_test):

        preds = self.predict(X_test)
        probs = self.predict_proba(X_test)

        print(classification_report(y_test, preds))
        print("ROC AUC:", roc_auc_score(y_test, probs))

        return preds, probs