import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from tensorflow.python.keras.utils.vis_utils import plot_model

from utils.confusion_matrix_pretty_print import plot_confusion_matrix_from_data


def plot_learn_model(model, img_file):
    plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True,
               dpi=96)


def pretty_plot_confusion_matrix(y_test, predictions):
    plot_confusion_matrix_from_data(y_test=y_test, predictions=predictions)


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=2.0)  # for label size
    sns.heatmap(cm, annot=True, annot_kws={"size": 20})  # font size
    plt.show()


def plot_ae_history(history):
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def scatter_data(data, indexes):
    plt.figure(figsize=(12, 12))
    plt.scatter(data[:, indexes[0]], data[:, indexes[1]])
    plt.show()


def scatter3d_data(data, indexes):
    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(data[:, indexes[0]], data[:, indexes[1]], data[:, indexes[2]])
    plt.title("simple 3D scatter plot")
    # show plot
    plt.show()


def visualization_2d(benign, gafgyt, mirai, indexes, s=20, fontsize=12):
    plt.figure(figsize=(12, 12))
    plt.scatter(benign[:, indexes[0]], benign[:, indexes[1]],
                label='Benign', s=s)
    plt.scatter(gafgyt[:, indexes[0]], gafgyt[:, indexes[1]],
                label='Gafgyt', s=s)
    plt.scatter(mirai[:, indexes[0]], mirai[:, indexes[1]],
                label='Mirai', s=s)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.title("Benign-Gafgyt-Mirai 2D Visualization")
    plt.show()


def visualization_3d(benign, gafgyt, mirai, indexes, s=20, fontsize=12):
    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(benign[:, indexes[0]], benign[:, indexes[1]], benign[:, indexes[2]],
                 label='Benign', s=s)
    ax.scatter3D(gafgyt[:, indexes[0]], gafgyt[:, indexes[1]], gafgyt[:, indexes[2]],
                 label='Gafgyt')
    ax.scatter3D(mirai[:, indexes[0]], mirai[:, indexes[1]], mirai[:, indexes[2]],
                 label='Mirai', s=s)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.title("Benign-Gafgyt-Mirai 3D Visualization")
    # show plot
    plt.show()


def visualizing_metric(metrics, pct, lim):
    density = gaussian_kde(metrics)
    plt.subplot(121)
    plt.axvline(x=pct, color='red')
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, density(xs))
    plt.title("Benign Metrics Percentile")
    plt.subplot(122)
    plt.axvline(x=lim, color='red')
    xs = np.linspace(0, 1, 200)
    plt.plot(xs, density(xs))
    plt.title("Benign Metrics S.D.")
    plt.show()


def quantization_errors_visualization(quantization_errors, error_treshold):
    plt.figure(figsize=(12, 8))
    plt.hist(quantization_errors)
    plt.axvline(error_treshold, color='k', linestyle='--')
    plt.xlabel('error')
    plt.ylabel('frequency')
    plt.show()


def outliers_visualization(data, is_outlier, indexes):
    plt.figure(figsize=(12, 12))
    plt.scatter(data[~is_outlier, indexes[0]], data[~is_outlier, indexes[1]],
                label='inlier')
    plt.scatter(data[is_outlier, indexes[0]], data[is_outlier, indexes[1]],
                label='outlier')
    plt.rc('legend', fontsize=20)
    plt.legend()
    # plt.savefig('resulting_images/som_outliers_detection.png')
    plt.show()


def plot_roc_curve_auc(fpr, tpr, roc_auc):
    plt.figure(figsize=(12, 8))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
