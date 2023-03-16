import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_predictions(predictions_df, t, figSize=(20, 10), show_only=-1):
    sns.set()

    fig = plt.figure(figsize=figSize)

    def get_style(c):
        if c == 'y_true':
            return 'x'
        if c == 'y_pred':
            return '*'
        else:
            return 's'

    samples = np.arange(0, predictions_df.shape[0] if show_only <= 0 else show_only)

    for column in predictions_df.columns:
        style = get_style(column)
        y_p = predictions_df[column].values
        plt.plot(samples, y_p[:len(samples)], style, label=f'{column} value')

    for sample in samples:
        plt.vlines(x=sample, ymin=np.min(predictions_df.iloc[sample][['y_lower', 'y_upper']].values),
                   ymax=np.max(predictions_df.iloc[sample][['y_lower', 'y_upper']].values), colors='cyan', alpha=0.3)
        plt.axvline(x=sample, color='b', alpha=0.002)

    if t:
        plt.axhline(y=t, color='r', alpha=0.4)

    plt.xlabel("samples")
    plt.ylabel("bit rate error")

    plt.legend(loc="upper left")
    plt.show()

    return fig
