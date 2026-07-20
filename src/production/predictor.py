# src/production/predictor.py

import joblib
import pandas as pd


class Predictor:

    def __init__(self, model_path, features_path):
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)

    def prepare_features(self, df):
        """
        Prima već filtrirani df (po strategiji i datumu)
        i samo izdvaja feature kolone
        """

        df = df.copy()

        # sigurnost: sort
        df = df.sort_index()

        # ako slučajno ima više redova po instrumentu → uzmi zadnji
        latest = df.groupby("Instrument").tail(1)

        # uzmi samo feature kolone
        X = latest[self.features]

        return X, latest

    def predict(self, df):
        """
        Glavna funkcija:
        - očekuje df za JEDAN datum (latest već filtriran u run_daily)
        """

        if df.empty:
            print("⚠️ Empty dataframe passed to predictor")
            return df

        X, latest = self.prepare_features(df)

        # provjera
        if len(X) == 0:
            print("⚠️ No features to predict")
            return latest

        # predikcija
        probs = self.model.predict_proba(X)[:, 1]

        latest = latest.copy()
        latest["prob"] = probs

        return latest