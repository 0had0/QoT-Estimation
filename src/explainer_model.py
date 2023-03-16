import pandas as pd
from shap import TreeExplainer, initjs
from xgboost.sklearn import XGBClassifier

initjs()


def shape_input(x, shape_values, prediction_interval, include_original_features, include_prediction_interval):
    if include_original_features:
        if include_prediction_interval and isinstance(prediction_interval, pd.DataFrame):
            return pd.concat([x, pd.DataFrame(shape_values, index=x.index), prediction_interval], axis=1).values
        return pd.concat([x, pd.DataFrame(shape_values, index=x.index)], axis=1).values
    return shape_values


class ExplanationClassifier:
    def __init__(self, model, include_original_features=False, include_prediction_interval=False):
        self.explainer = TreeExplainer(model=model)
        self.model = XGBClassifier()
        self.include_original_features = include_original_features
        self.include_prediction_interval = include_prediction_interval

    def fit(self, x, y, prediction_interval=None):
        assert len(x) == len(y)
        # assert not self.include_prediction_interval and not prediction_interval
        self.model.fit(shape_input(x, self.shape_values(x), prediction_interval, self.include_original_features,
                                   self.include_prediction_interval), y)

    def predict(self, x, prediction_interval=None):
        # assert not self.include_prediction_interval and not prediction_interval
        return self.model.predict(
            shape_input(x, self.shape_values(x), prediction_interval, self.include_original_features,
                        self.include_prediction_interval))

    def shape_values(self, x):
        return self.explainer.shap_values(x)

    def generate_shape_values_df(self, x, label):
        assert len(x) == len(label)
        return pd.concat([self.shape_values(x), pd.DataFrame({"label": label})])
