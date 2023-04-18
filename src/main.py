import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import shap

from prediction_uncertainty_model import PredictionIntervalModel
from explainer_model import ExplanationClassifier

from utils import get_crossing_threshold, get_data, calculate_th, get_data_of_second_dataset, get_true_not_crossing_thrushold, get_false_not_crossing_thrushold


class Model:
    def __init__(self, th, include_original_features=False, include_prediction_interval=False):
        self.pi_model = PredictionIntervalModel(t=th)
        self.explainer_classifier = None
        self.pi_prediction_df = None
        self.th = th
        self.include_original_features = include_original_features
        self.include_prediction_interval = include_prediction_interval
        self.output = None

    def fit(self, x, y):
        x_pi, x_exp, y_pi, y_exp = train_test_split(x, y, test_size=0.5, random_state=75)
        self.pi_model.fit(x_pi, y_pi)
        self.pi_prediction_df = self.pi_model.prediction_df(x_exp, y_exp)

        crossing_outcome = get_crossing_threshold(self.pi_prediction_df, self.th)

        self.explainer_classifier = ExplanationClassifier(model=self.pi_model.models['y_pred'],
                                                          include_original_features=self.include_original_features,
                                                          include_prediction_interval=self.include_prediction_interval)
        self.explainer_classifier.fit(
            x_exp[x_exp.index.isin(crossing_outcome.index)],
            ((crossing_outcome['y_pred'] < self.th).astype(int).values == (crossing_outcome['y_true'] < self.th).astype(
                int).values).astype(int),
            crossing_outcome[['y_lower', 'y_upper']] if self.include_prediction_interval else None
        )

    def predict(self, x):
        out = self.pi_model.predict(x)
        out.index = x.index

        out['is_uncertain'] = ((out['y_lower'] < self.th) & (self.th < out['y_upper'])).astype(int)

        uncertain_outcome = out[out['is_uncertain'] == 1]

        validation_outcome = self.explainer_classifier.predict(
            x[x.index.isin(uncertain_outcome.index)],
            uncertain_outcome[['y_lower',
                               'y_upper']] if self.include_prediction_interval else None)

        out['is_valid_prediction'] = np.ones(len(out.index)).astype(int)

        out.loc[uncertain_outcome.index, 'is_valid_prediction'] = validation_outcome

        out['class_pred'] = (out['y_pred'] < self.th).astype(int)

        out['corrected_prediction'] = [
            class_pred if is_valid_prediction else int(not bool(class_pred))
            for is_valid_prediction, class_pred in zip(out['is_valid_prediction'], out['class_pred'])
        ]

        self.output = out

        return out['corrected_prediction'].values, out

    def evaluate_model_1(self, y):
        y_uncertain = self.output[self.output['is_uncertain'] == 1]['corrected_prediction']
        y_certain = self.output[self.output['is_uncertain'] != 1]['corrected_prediction']

        print(
            f'[Model 1 evaluation]: {len(y_certain) / len(self.output) * 100:.4}% certain outcome, '
            f'{len(y_uncertain) / len(self.output) * 100:.4}% uncertain outcome'
        )

        y_t = y[y.index.isin(y_certain.index)]

        print(confusion_matrix(y_t, y_certain.values))
        print(classification_report(y_t, y_certain.values))

    def evaluate_model_2(self, y):
        y_predicted = self.output[self.output['is_uncertain'] == 1]['corrected_prediction'].values
        y_t = y[y.index.isin(self.output[self.output['is_uncertain'] == 1].index)]
        print(confusion_matrix(y_t, y_predicted))
        print(classification_report(y_t, y_predicted, digits=4))


if __name__ == '__main__':
    # data, target = get_data()
    data, target = get_data_of_second_dataset()
    X_train, X_test, y_train, y_test = train_test_split(data, target['ber'], test_size=0.20, random_state=75)

    threshold = calculate_th(target[['class', 'ber']], 'ber')

    model = Model(th=threshold)
    # model_1 = Model(th=threshold, include_original_features=True)
    # model_2 = Model(th=threshold, include_original_features=True, include_prediction_interval=True)
    # model_3 = Model(th=threshold, include_prediction_interval=True)

    model.fit(X_train, y_train)
    # model_1.fit(X_train, y_train)
    # model_2.fit(X_train, y_train)
    # model_3.fit(X_train, y_train)

    y_pred, out = model.predict(X_test)
    # y_pred_1, _ = model_1.predict(X_test)
    # y_pred_2, _ = model_2.predict(X_test)
    # y_pred_3, _ = model_3.predict(X_test)

    y_true = target.loc[y_test.index, 'class']

    # print('Uncertainty detection + correction (shap values only)')
    # model.evaluate_model_2(y_true)
    #
    # print('Uncertainty detection + correction (shap values + data features)')
    # model_1.evaluate_model_2(y_true)
    #
    # print('Uncertainty detection + correction (shap values + data features + prediction interval)')
    # model_2.evaluate_model_2(y_true)
    #
    # print('Uncertainty detection + correction (shap values + prediction interval)')
    # model_3.evaluate_model_2(y_true)
    #
    # print("base model")

    # y_pred_base = (
    #         out['y_pred'] < threshold
    # ).astype(int)
    #
    # print(confusion_matrix(y_true, y_pred_base))
    # print(classification_report(y_true, y_pred_base))

    # model.evaluate_model_1(y_true.astype(int))

    y_uncertain = out[out['is_uncertain'] == 1]
    y_certain = out[~out.index.isin(y_uncertain.index)]['class_pred']

    assert len(y_uncertain) + len(y_certain) == len(out)

    y_t = y_true[y_true.index.isin(y_certain.index)].values.astype(int)
    y_p = y_certain.values.astype(int)

    print(np.unique(y_t, return_counts=True))
    print(np.unique(y_p, return_counts=True))

    print(confusion_matrix(y_t, y_p))
    print(classification_report(y_t, y_p))
