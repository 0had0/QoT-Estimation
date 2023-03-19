import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import get_data


def get_tuned_models():
    quantiles_params = pickle.load(open('./cache/gradient_boost_q_params.pkl', 'rb'))
    ms = {
        'y_lower' if float(a) == np.array(list(quantiles_params.keys())).astype(float).min()
        else 'y_upper': HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(a),
            random_state=75,
            **param
        )
        for a, param in quantiles_params.items()
    }

    ms['y_pred'] = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=75,
        **pickle.load(open('./cache/gradient_boost_se_params.pkl', 'rb'))
    )

    return ms


class PredictionIntervalModel:
    def __init__(self, t):
        self.th = t
        self.models = {
            str(key): m
            for key, m in get_tuned_models().items()
        }

    def fit(self, x, y):
        self.models = {
            str(key): m.fit(x, y)
            for key, m in self.models.items()
        }

    def predict(self, x):
        return pd.DataFrame({
            str(key): m.predict(x)
            for key, m in self.models.items()
        })

    def prediction_df(self, x, y):
        prediction_df = self.predict(x)
        prediction_df.index = y.index
        prediction_df['y_true'] = y

        return prediction_df


def create_models(aas: [float, float]) -> dict:
    return {
        str(a): HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(a),
            early_stopping=True,
            verbose=2
        ) for a in aas}


if __name__ == '__main__':
    alphas = [0.025, 0.975]
    param_grid = dict(
        learning_rate=[0.05, 0.1, 0.2],
        max_depth=[2, 5, 10],
        max_iter=[50, 100, 150],
        min_samples_leaf=[1, 5, 10, 20],
    )

    params = {}

    X_train, y_train = get_data()

    models = create_models(alphas)

    for alpha, model in models.items():
        search = GridSearchCV(
            model,
            param_grid,
            verbose=1
        ).fit(X_train, y_train['ber'].values)

        params[str(alpha)] = search.best_params_

    pickle.dump(params, open('./cache/gradient_boost_q_params.pkl', 'wb'))

    se_search = GridSearchCV(
        HistGradientBoostingRegressor(loss="squared_error", verbose=2, early_stopping=True),
        param_grid,
        verbose=1
    ).fit(X_train, y_train['ber'].values)

    pickle.dump(se_search.best_params_, open('./cache/gradient_boost_se_params.pkl', 'wb'))

