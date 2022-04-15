# visualization
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import STATUS_OK
from pylab import plot, axis, show, pcolor, colorbar, bone

from utils import minisom
from utils.config import markers, colors


def plot_som(som, data, target=None):
    plt.figure(figsize=(16, 12))
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    if target is not None:
        for cnt, xx in enumerate(data):
            w = som.winner(xx)
            plot(w[0] + .5, w[1] + .5, markers[target[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[target[cnt]], markersize=12, markeredgewidth=2)
            axis([0, som._weights.shape[0], 0, som._weights.shape[1]])
        show()
    else:
        for cnt, xx in enumerate(data):
            w = som.winner(xx)
            plot(w[0] + .5, w[1] + .5, markers[0], markerfacecolor='None',
                 markeredgecolor=colors[0], markersize=12, markeredgewidth=2)
        show()


def som_fn(space):
    # print("Hyper-parameters optimization in function som_fn")
    sig = space['sigma']
    learning_rate = space['learning_rate']
    x = int(space['x'])
    data_benign = space['data_benign']
    val = minisom.MiniSom(x=x,
                          y=x,
                          input_len=data_benign.shape[1],
                          sigma=sig,
                          learning_rate=learning_rate,
                          ).quantization_error(data_benign[0:100, :])
    # print(space)
    # print("Current value {}".format(val))
    return {'loss': val, 'status': STATUS_OK}


def get_anomalies(benign_metrics, anomaly_metrics, alpha=3, return_outliers=True):
    '''
    Get anomalies from evaluation metric
    '''
    limit = np.mean(benign_metrics) + np.std(benign_metrics) * alpha
    outliers = np.argwhere(np.abs(anomaly_metrics) > limit)
    print("Determined: ", len(outliers) / len(anomaly_metrics) * 100, "% as anomaly")
    if return_outliers:
        return outliers


def minimize_anomaly(benign_metrics, anomaly_metrics, alpha=3):
    '''
    Objective function to be minimized durinig tuning
    calculates percent error in classifying anomalies based on steady state metrics.
    '''
    limit = np.mean(benign_metrics) + np.std(benign_metrics) * alpha
    outliers = np.argwhere(np.abs(anomaly_metrics) > limit)
    pct_anomaly = len(outliers) / len(anomaly_metrics)
    return 1 - pct_anomaly


