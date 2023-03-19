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
    def __init__(self, model, include_original_features=False, include_prediction_interval=False, xgb_clf=None):
        self.explainer = TreeExplainer(model=model)
        self.model = xgb_clf if xgb_clf is not None else XGBClassifier()
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


if __name__ == "__main__":
    from prediction_uncertainty_model import PredictionIntervalModel
    from utils import get_crossing_threshold, get_data, calculate_th
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    import numpy as np
    import pickle
    from sklearn.metrics import confusion_matrix, classification_report

    # <==== Need to change to a static input

    data, target = get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, target['ber'], test_size=0.05, random_state=75)

    threshold = calculate_th(target[['class', 'ber']], 'ber')

    x_pi, x_exp, y_pi, y_exp = train_test_split(X_train, y_train, test_size=0.5, random_state=75)

    pi_model = PredictionIntervalModel(t=threshold)
    pi_model.fit(x_pi, y_pi)
    pi_prediction_df = pi_model.prediction_df(x_exp, y_exp)

    crossing_outcome = get_crossing_threshold(pi_prediction_df, threshold)

    x, y = (x_exp[x_exp.index.isin(crossing_outcome.index)],
            ((crossing_outcome['y_pred'] < threshold).astype(int).values == (
                    crossing_outcome['y_true'] < threshold).astype(
                int).values).astype(int))

    print(np.unique(y, return_counts=True))

    # Need to change to a static input ====>

    # xgb_model = XGBClassifier()
    #
    # params = {'max_depth': [15, 20, 25],
    #           'learning_rate': [0.01, 0.1, 0.2],
    #           'subsample': np.arange(0.5, 1.0, 0.1),
    #           'n_estimators': [250, 500, 750],
    #           }
    #
    # clf = RandomizedSearchCV(estimator=xgb_model,
    #                          param_distributions=params,
    #                          scoring='accuracy',
    #                          n_iter=25,
    #                          n_jobs=4,
    #                          verbose=1)
    #
    # clf.fit(x, y)

    best_combination = pickle.load(open('cache/xgb_classifier_hyper_params', 'rb'))
    # best_combination = clf.best_params_

    # pickle.dump(best_combination, open('cache/xgb_classifier_hyper_params', 'wb'))

    y_true = target.loc[y_test.index, 'class'].values


    def evaluate(model):
        model.fit(x, y)
        y_pred = model.predict(X_test)

        cf_matrix = confusion_matrix(y_true, y_pred).ravel()
        print(confusion_matrix(y_true, y_pred))
        for val, key in zip(cf_matrix, ['tn', 'fp', 'fn', 'tp']):
            print(f'{key}: {val}')
        print(classification_report(y_true, y_pred))


    # Old model
    evaluate(ExplanationClassifier(model=pi_model.models['y_pred']))

    # New model
    evaluate(ExplanationClassifier(model=pi_model.models['y_pred'], xgb_clf=XGBClassifier(**best_combination)))
