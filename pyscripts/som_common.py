import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import somoclu
from hyperopt import hp, Trials, fmin, tpe, rand, atpe, anneal
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

from utils import minisom
from utils.config import colors
from utils.som_utils import plot_som
from utils.som_utils import som_fn
from utils.visualization import quantization_errors_visualization, outliers_visualization, plot_confusion_matrix, \
    pretty_plot_confusion_matrix, plot_roc_curve_auc


def _som_classify(som, winmap, data):  # , X_train, y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    # winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result


def _som_clustering(som, data, cluster_index):
    plt.figure(figsize=(20, 20))

    # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(cluster_index):
        plt.scatter(data[cluster_index == c, 0],
                    data[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

    # plotting centroids
    for centroid in som.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                    s=80, linewidths=35, color='k', label='centroid')
    plt.legend()
    plt.show()


def som_classification(som, winmap, X_test, y_test):
    '''
    Phan loai cho bengin va cac lop khac.
    '''
    # xu ly cho truong hop 1, tinh lai confusion matrix, benign la lop 1, con lai la lop 2
    # y_train = [i if i == 1 else 2 for i in y_train]
    # chuyen doi label cho y_test
    y_test = [i if i == 1 else 2 for i in y_test]
    y_pred = _som_classify(som=som, winmap=winmap, data=X_test)  # , X_train=X_train, y_train=y_train)
    # chuyen doi label cho y_pred
    y_pred = [i if i == 1 else 2 for i in y_pred]
    print(classification_report(y_test, y_pred, digits=3))
    # pretty_plot_confusion_matrix(y_test=y_test, predictions=y_pred)
    # print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm=cm)
    # plot roc curve and calculate auc_score
    # chi su dung cho tu hai lop tro len
    if len(np.unique(y_test)) >= 2 and len(np.unique(y_pred)) >= 2:
        fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=2)
        auc_score = roc_auc_score(y_test, y_pred)
        print("AUC score: ", auc_score)
        # roc_auc = auc(fpr, tpr)
        # print(roc_auc)
        plot_roc_curve_auc(fpr=fpr, tpr=tpr, roc_auc=auc_score)


def _grid_search(params):
    gs = GridSearchCV()

def train_som(X_train, y_train, algo='tpe', som_x=None, som_y=None, sigma=6, learning_rate=2.0, verbose=False,
              show_progressbar=False, max_evals=500):
    print("---------------------------------Train SOM-------------------------------------")
    print("Number of feature: ", X_train.shape[1])
    # TRAINING AND TUNING PARAMS FOR SOM
    # get parameters
    # the params haven't passes --> optimize to get best values
    if som_x is None and som_y is None:
        print("The default values of som_x and som_y are None")
        print("Hyper-parameters optimization process. The algorithm used is {}.".format(algo))
        if algo.__eq__('tpe'):
            algo = tpe.suggest
        elif algo.__eq__('rand'):
            algo = rand.suggest
        elif algo.__eq__('atpe'):
            algo = atpe.suggest
        elif algo.__eq__('anneal'):
            algo = anneal.suggest
        else:
            print("Default algorithm is tpe")
            algo = tpe.suggest
        space = {
            'sigma': hp.uniform('sigma', 5, 10),
            'learning_rate': hp.uniform('learning_rate', 0.05, 5),
            'x': hp.uniform('x', 20, 50),
            'data_benign': X_train
        }
        trials = Trials()
        # max_evals can be set to 1200, but for speed, we set to 100
        best = fmin(fn=som_fn,
                    space=space,
                    algo=algo,
                    max_evals=max_evals,
                    trials=trials,
                    verbose=verbose,
                    show_progressbar=show_progressbar)
        print('Best: {}'.format(best))
        som_x = math.ceil(best['x'])
        som_y = math.ceil(best['x'])
        sigma = best['sigma']
        learning_rate = best['learning_rate']
    som_turned = minisom.MiniSom(x=som_x,
                                 y=som_y,
                                 input_len=X_train.shape[1],
                                 sigma=sigma,
                                 learning_rate=learning_rate)
    # som_turned = minisom.MiniSom(x=40,
    #                              y=40,
    #                              input_len=X_train_normalized.shape[1],
    #                              sigma=6,
    #                              learning_rate=2.0)
    print("---------SOM has been turned!-----------")
    print("Starting SOM Weights init")
    som_turned.pca_weights_init(X_train)
    print("Perform SOM (turned) train random")
    som_turned.train_random(X_train, 1000, verbose=0)
    winmap = som_turned.labels_map(X_train, y_train)
    y_benign = [y for y in y_train if y == 1]
    outliers_percentage = len(y_benign) / len(y_train)
    print(outliers_percentage)
    return som_turned, winmap, outliers_percentage


def test_som(som_turned, winmap, outliers_percentage, X_test, y_test):
    #     # kiem tra thu voi du lieu huan luyen
    #     quantization_errors = np.linalg.norm(som_turned.quantization(X_train_normalized) - X_train_normalized, axis=1)
    # kiem tra thu voi du lieu test
    print("----------------------------------------------------------------------")
    print("Compute quantization errors and error threshold")
    quantization_errors = np.linalg.norm(som_turned.quantization(X_test) - X_test, axis=1)
    print(quantization_errors)
    error_treshold = np.percentile(quantization_errors,
                                   min(100 * (1 - outliers_percentage) + 5, 100))
    is_outlier = quantization_errors > error_treshold

    print("Visualize quantization error")
    quantization_errors_visualization(quantization_errors=quantization_errors, error_treshold=error_treshold)

    print("Outliers visualization")
    #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[0, 1])
    #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[1, 2])
    outliers_visualization(data=X_test, is_outlier=is_outlier, indexes=[0, 1])
    outliers_visualization(data=X_test, is_outlier=is_outlier, indexes=[1, 2])

    print("----------------------------------------------------------------------")
    print("SOM classification")
    # som_classification(som=som_turned, X_train=X_train_normalized, y_train=y_train, X_test=X_test_normalized, y_test=y_test)
    som_classification(som=som_turned, winmap=winmap, X_test=X_test, y_test=y_test)
    print("-----------Testing SOM done!-------------")


def process_som(X_train_normalized, y_train, X_test_normalized, y_test):
    # # tam thoi ngat bo cai plot_som de tang toc
    # n = 5000
    # x = int(1 / 2 * np.sqrt(n))
    # som_og = minisom.MiniSom(x=x,
    #                          y=x,
    #                          input_len=X_train_normalized.shape[1],
    #                          sigma=2,
    #                          learning_rate=0.5)
    # som_og.train_random(X_train_normalized, 100)
    # # plot som
    # plot_som(som_og, X_train_normalized[0:n, :], y_train[0:n])
    # plot_som(som_og, X_test_normalized[0:n, :], y_test[0:n])

    # data_benign_new = normalize(data_benign_train_scaled)
    # data_gafgyt_combo_train_new = normalize(data_gafgyt_combo_train)
    # data_gafgyt_junk_train_new = normalize(data_gafgyt_junk_train)
    # data_gafgyt_tcp_train_new = normalize(data_gafgyt_tcp_train)
    # data_gafgyt_udp_train_new = normalize(data_gafgyt_udp_train)
    # data_gafgyt_scan_train_new = normalize(data_gafgyt_scan_train)
    #
    # data_mirai_ack_new = normalize(data_mirai_ack_train)
    # data_mirai_scan_new = normalize(data_mirai_scan_train)
    # data_mirai_syn_new = normalize(data_syn)
    # data_mirai_udp_new = normalize(data_udp)
    # data_mirai_udpplain_new = normalize(data_mirai_udpplain_train)
    # data_new = normalize(data)

    # --------------------Region SOM---------------------------#
    print("----------------------------------------------------------------------")
    print("Number of feature: ", X_train_normalized.shape[1])
    # TRAINING AND TUNING PARAMS FOR SOM
    # get parameters
    print("Hyper-parameters optimization process")
    space = {
        'sigma': hp.uniform('sigma', 5, 10),
        'learning_rate': hp.uniform('learning_rate', 0.05, 5),
        'x': hp.uniform('x', 20, 50),
        'data_benign': X_train_normalized
    }
    trials = Trials()
    # max_evals can be set to 1200, but for speed, we set to 100
    best = fmin(fn=som_fn,
                space=space,
                algo=tpe.suggest,
                max_evals=500,
                trials=trials,
                verbose=False,
                show_progressbar=False)
    print('Best: {}'.format(best))
    som_turned = minisom.MiniSom(x=math.ceil(best['x']),
                                 y=math.ceil(best['x']),
                                 input_len=X_train_normalized.shape[1],
                                 sigma=best['sigma'],
                                 learning_rate=best['learning_rate'])
    # som_turned = minisom.MiniSom(x=40,
    #                              y=40,
    #                              input_len=X_train_normalized.shape[1],
    #                              sigma=6,
    #                              learning_rate=2.0)
    print("SOM has been turned!")
    print("SOM Weights init")
    som_turned.pca_weights_init(X_train_normalized)
    print("SOM turned train random")
    som_turned.train_random(X_train_normalized, 1000, verbose=0)
    # n = 10000
    # plot_som(som_turned, X_train_normalized[0:n, :], y_train[0:n])
    y_benign = [y for y in y_train if y == 1]
    outliers_percentage = len(y_benign) / len(y_train)
    print(outliers_percentage)
    #     # kiem tra thu voi du lieu huan luyen
    #     quantization_errors = np.linalg.norm(som_turned.quantization(X_train_normalized) - X_train_normalized, axis=1)
    # kiem tra thu voi du lieu test
    print("----------------------------------------------------------------------")
    print("Compute quantization errors and error threshold")
    quantization_errors = np.linalg.norm(som_turned.quantization(X_test_normalized) - X_test_normalized, axis=1)
    print(quantization_errors)
    error_treshold = np.percentile(quantization_errors,
                                   min(100 * (1 - outliers_percentage) + 5, 100))
    is_outlier = quantization_errors > error_treshold

    print("Visualize quantization error")
    quantization_errors_visualization(quantization_errors=quantization_errors, error_treshold=error_treshold)

    print("Outliers visualization")
    #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[0, 1])
    #     outliers_visualization(data=X_train_normalized, is_outlier=is_outlier, indexes=[1, 2])
    outliers_visualization(data=X_test_normalized, is_outlier=is_outlier, indexes=[0, 1])
    outliers_visualization(data=X_test_normalized, is_outlier=is_outlier, indexes=[1, 2])

    print("----------------------------------------------------------------------")
    print("SOM classification")
    som_classification(som=som_turned, X_train=X_train_normalized, y_train=y_train, X_test=X_test_normalized,
                       y_test=y_test)

    print("----------------------------------------------------------------------")
    print("SOM turned clustering")
    som_shape = (1, 6)
    som_turned = minisom.MiniSom(som_shape[0], som_shape[1], X_train_normalized.shape[1], sigma=0.5, learning_rate=0.5,
                                 neighborhood_function='gaussian', random_seed=10)
    # each neuron represents a cluster
    print("Clustering training data")
    winner_coordinates = np.array([som_turned.winner(x) for x in X_train_normalized]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    _som_clustering(som=som_turned, data=X_train_normalized, cluster_index=cluster_index)

    print("Clustering testing data")
    winner_coordinates = np.array([som_turned.winner(x) for x in X_test_normalized]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    _som_clustering(som=som_turned, data=X_test_normalized, cluster_index=cluster_index)

    # Somoclu representation
    print("SOM somoclu visualization")
    som_somoclu = somoclu.Somoclu(n_columns=math.ceil(best['x']),
                                  n_rows=math.ceil(best['x']),
                                  std_coeff=best['sigma'])
    som_somoclu.train(X_train_normalized[0:1000, :],
                      scale0=0.2,
                      scaleN=0.02)
    # colors = ['red', 'green', 'blue', 'pink', 'yellow']
    color_list = [colors[t] for t in y_train[0:1000]]
    som_somoclu.view_umatrix(bestmatches=True, bestmatchcolors=color_list)
