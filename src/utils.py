import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler


def preprocess(d, t):
    scaler = StandardScaler()
    d1 = d.drop(["conn_id", "src_id", "dst_id"], axis=1)
    t1 = t.drop(["osnr", "snr"], axis=1)
    return (pd.DataFrame(scaler.fit_transform(d1), columns=d1.columns, index=d1.index),
            t1)


def load_dataset(path: str) -> (pd.DataFrame, pd.DataFrame):
    lp_ds = xr.open_dataset(path)
    data = lp_ds.data.to_pandas()
    target = lp_ds.target.to_pandas()

    return data, target


def get_data():
    d, t = split_on_mod_order_16(*load_dataset('../datasets/lightpath_dataset_1.nc'))
    return preprocess(d[0], t[0])


def get_data_of_second_dataset():
    return preprocess(*load_dataset('../datasets/lightpath_dataset_4.nc'))


def split_on_mod_order_16(d: pd.DataFrame, t: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    is_mod_order_16 = (d['mod_order'] == 16.0)
    return (
        (d[~is_mod_order_16], d[is_mod_order_16]),
        (t[~is_mod_order_16], t[is_mod_order_16])
    )


def get_class(s: pd.Series, t: float):
    return (s < t).astype(int)


def get_crossing_threshold(df: pd.DataFrame, th: float) -> pd.DataFrame:
    return df[(df['y_lower'] < th) & (th < df['y_upper'])]


def is_true_prediction(df: pd.DataFrame, th: float):
    return get_class(df['y_pred'], th) == get_class(df['y_true'], th)


def is_false_prediction(df: pd.DataFrame, th: float):
    return get_class(df['y_pred'], th) != get_class(df['y_true'], th)


def get_true_not_crossing_thrushold(df: pd.DataFrame, th: float) -> pd.DataFrame:
    crossing_thrushold = get_crossing_threshold(df, th)
    return df[~df.index.isin(crossing_thrushold.index) & (is_true_prediction(df, th))]


def get_false_not_crossing_thrushold(df: pd.DataFrame, th: float) -> pd.DataFrame:
    crossing_thrushold = get_crossing_threshold(df, th)
    return df[~df.index.isin(crossing_thrushold.index) & (is_false_prediction(df, th))]


def calculate_th(data: pd.DataFrame, feature: str):
    return np.mean([
        np.max(data[feature][data['class'] == 1.0].values),
        np.min(data[feature][data['class'] == 0.0].values)
    ])
