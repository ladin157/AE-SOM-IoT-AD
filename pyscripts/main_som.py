import gc
import time

import numpy as np
from hyperopt import tpe
from sklearn.model_selection import train_test_split

from ad.algo.ae import ae_process, denoising_ae_process
from ad.algo.tree import tree_process
from ad.algo.pca import pca_process
from pyscripts.som_common import process_som, train_som, test_som
from utils.config import nbaiot_1K_data_path
from utils.datasets import get_train_test_data, get_test_data
from utils.preprocessing import scale_data, normalize_data

'''
Data preprocessing:
1. Chi thu voi benign va mirai
    Gafgyt co the dung nhu du lieu testing
2. Chi thu voi benign va gafgyt
Flow:
1. Load data
    - Combine data
    - Get train, test data
2. Scale data
3. Normalize data
4. Train SOM
5. Optimize SOM
6. Clustering
7. Classify

AE Flow:
1. Load data
2. Scale data
3. Train AE
4. Get latent representation
5. Normalize data
6. Train SOM
7. Optimize SOM
8. Clustering
9. Classify

Tree Flow:
1. Load data
2. Scale data
3. Train tree
4. Get features (selection)
5. Normalize data
6. Train SOM
7. Optimize SOM
8. Clustering
9. Classify

'''


def som_test(som, winmap, outliers_percentage, scaler, X_test, y_test, encoder=None, pca=None):
    # scal data
    print("Shape: ", X_test.shape, y_test.shape)
    print("----------------------Test is starting----------------------")
    print("Scale data")
    X_test = scaler.transform(X_test)
    print("Shape: ", X_test.shape, y_test.shape)
    if encoder is not None:
        print("Encode data using trained encoder")
        X_test = encoder.predict(X_test)
    if pca is not None:
        print("Encode data using trained tree")
        X_test = pca.transform(X_test)
    print("Shape: ", X_test.shape, y_test.shape)
    # normalize data
    print("Normalize data")
    _, X_test = normalize_data(None, X_test)
    print("Shape: ", X_test.shape, y_test.shape)
    print("Testing")
    test_som(som_turned=som, winmap=winmap, outliers_percentage=outliers_percentage, X_test=X_test,
             y_test=y_test)
    print("----------------------Test Done----------------------")


def main_v01(choose_folder, choose_index, method='som', train_size=1.0, test_size=1.0, gafgyt=False,
             test_gafgyt_on_mirai=False):
    '''
    :param choose_folder: The folder to load data.
    :param choose_index: The device type to test.
    :param method: The method used to prepare data. It can be 'som', 'ae', or 'tree'.
    :param train_size: The train_sizex to set in function train_test_split (default is 1, and only apply other values on full data).
    :param test_size: The test_size to split the test dataset
    :param gafgyt: test for gafgyt attack type.
    :param test_gafgyt_on_mirai: Train on mirai, but use gafgyt for testing.
    '''
    print("===========================Start=====================================")
    start_time = time.time()
    print("======================================================================")
    if gafgyt:
        print("Training and testing on gafgyt data")
    else:
        print("Traing and testing on mirai data")
    # get train and test data
    X_train, y_train, X_test, y_test, X_gafgyt, y_gafgyt = get_train_test_data(choose_folder=choose_folder,
                                                                               choose_index=choose_index, gafgyt=gafgyt)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=1, shuffle=True)
    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        if X_gafgyt is not None and y_gafgyt is not None:
            X_gafgyt, _, y_gafgyt, _ = train_test_split(X_gafgyt, y_gafgyt, test_size=train_size, random_state=1,
                                                        shuffle=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # scale data from initial data
    print("======================================================================")
    print("Scale data")
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train=X_train, X_test=X_test)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)

    # #     data = resample(data)
    # data_benign_train_scaled = scaler.transform(data_benign)
    # data_train_scaled = scaler.transform(data_train)
    # data_test_scaled = scaler.transform(data_test)
    # # for other
    # data_gafgyt_combo_train_scaled = scaler.transform(data_gafgyt_combo_train)
    # data_gafgyt_junk_train_scaled = scaler.transform(data_gafgyt_junk_train)
    # data_gafgyt_scan_train_scaled = scaler.transform(data_gafgyt_scan_train)
    # data_gafgyt_tcp_train_scaled = scaler.transform(data_gafgyt_tcp_train)
    # data_gafgyt_udp_train_scaled = scaler.transform(data_gafgyt_udp_train)
    #
    # data_mirai_ack_train_scaled = scaler.transform(data_mirai_ack_train)
    # data_mirai_scan_train_scaled = scaler.transform(data_mirai_scan_train)
    # data_mirai_syn_train_scaled = scaler.transform(data_mirai_syn_train)
    # data_mirai_udp_train_scaled = scaler.transform(data_mirai_udp_train)
    # data_mirai_udpplain_train_scaled = scaler.transform(data_mirai_udpplain_train)

    ## With SOM --> train on Benign --> test on other
    # creating and training SOM
    # n = 5000
    # x = int(1 / 2 * np.sqrt(n))
    # som_og = minisom.MiniSom(x=x,
    #                          y=x,
    #                          input_len=X_train_scaled.shape[1],
    #                          sigma=2,
    #                          learning_rate=0.5)
    # som_og.train_random(X_train_scaled, 100)
    # # plot som
    # plot_som(som_og, X_train_scaled[0:n, :], y_train[0:n])
    # plot_som(som_og, X_test_scaled[0:n, :], y_test[0:n])

    # # normalize data
    print("=================================Method {}=====================================".format(method))
    if method.__eq__('ae'):
        print("AE process")
        X_train_scaled, X_test_scaled, encoder = ae_process(X=X_train_scaled, X_test=X_test_scaled)
    elif method.__eq__('tree'):
        print("Tree process")
        X_train_scaled, X_test_scaled = tree_process(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)

    print("======================================================================")
    print("Normalize data")
    # normalize data from scaled data
    X_train_normalized, X_test_normalized = normalize_data(X_train=X_train_scaled, X_test=X_test_scaled)
    print(X_train_normalized.shape, y_train.shape, X_test_normalized.shape, y_test.shape)

    # process SOM
    print("======================================================================")
    process_som(X_train_normalized=X_train_normalized, y_train=y_train, X_test_normalized=X_test_normalized,
                y_test=y_test)

    # if test_with_gafgyt on mirai is True --> test on gafgy

    if test_gafgyt_on_mirai:
        print("======================================================================")
        print("Test gafgyt data on mirai train")
        # scale data
        print("Scale data")
        X_gafgyt = scaler.transform(X_gafgyt)
        print(X_gafgyt.shape)
        # encode data
        print("Encode data")
        X_gafgyt = encoder.predict(X_gafgyt)
        print(X_gafgyt.shape)
        # normalize data
        print("Normalize data")
        X_gafgyt, _ = normalize_data(X_gafgyt, None)
        print(X_gafgyt.shape)
        process_som(X_train_normalized=X_train_normalized, y_train=y_train, X_test_normalized=X_gafgyt, y_test=y_gafgyt)

    del X_train
    del X_train_scaled
    del X_train_normalized
    del y_train
    del X_test
    del X_test_scaled
    del X_test_normalized
    del X_gafgyt
    del y_gafgyt
    gc.collect()
    end_time = time.time()
    print('Total time: {}'.format(end_time - start_time))
    # print(end_time - start_time)
    # ## Anomalies detection ##
    # training = data_benign_new
    # evaluation = data_new
    #
    # # initialize our anomaly detector with some arbitrary parameters
    # anomaly_detector = AnomalyDetection(shape=(10, 10),
    #                                     input_size=training.shape[1],
    #                                     learning_rate=8,
    #                                     learning_decay=0.001,
    #                                     initial_radius=2,
    #                                     radius_decay=0.001,
    #                                     min_number_per_bmu=0,
    #                                     number_of_neighbors=3)
    # # fit the anomaly detector and apply to the evaluation data
    # anomaly_detector.fit(training, 5000)
    # # set limits
    # benign_metrics = anomaly_detector.evaluate(data_benign_new)
    # alpha = 3
    #
    # sd_benign = np.std(benign_metrics)
    # mean_benign = np.mean(benign_metrics)
    # lim_benign = mean_benign + alpha * sd_benign
    # pct_benign = np.percentile(benign_metrics, 99.7)
    # print("sd_benign: {}, mean_benign: {}, lim_benign: {}, pct_benign: {}".format(sd_benign, mean_benign, lim_benign,
    #                                                                               pct_benign))
    # # Visualize benign metric
    # visualizing_metric(metrics=benign_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualize all anomlies
    # anomaly_metrics = anomaly_detector.evaluate(evaluation)
    # visualizing_metric(metrics=anomaly_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing ack attack
    # ack_metrics = anomaly_detector.evaluate(data_ack_new)
    # visualizing_metric(metrics=ack_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing scan attack
    # scan_metrics = anomaly_detector.evaluate(data_scan_new)
    # visualizing_metric(metrics=scan_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing syn attack
    # syn_metrics = anomaly_detector.evaluate(data_syn_new)
    # visualizing_metric(metrics=syn_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing udp attack
    # udp_metrics = anomaly_detector.evaluate(data_udp_new)
    # visualizing_metric(metrics=udp_metrics, lim=lim_benign, pct=pct_benign)
    #
    # # Anomaly detection
    # print("Anomaly detection")
    # metrics = [benign_metrics, anomaly_metrics, ack_metrics, scan_metrics, syn_metrics, udp_metrics]
    # metric_names = ['benign', 'all alnomalies', 'ack', 'scan', 'syn', 'udp']
    # alpha = 3
    # for metric, name in zip(metrics, metric_names):
    #     print(name)
    #     get_anomalies(benign_metrics, metric, alpha, False)


def load_common_data(choose_folder, train_index, train_size=1.0, test_size=1.0, gafgyt=False):
    # get train and test data
    X_train, y_train, X_test, y_test, X_train_benign, y_train_benign, X_test_benign, y_test_benign = get_train_test_data(
        choose_folder=choose_folder,
        choose_index=train_index, gafgyt=gafgyt)
    print(X_train_benign.shape, y_train_benign.shape, X_test_benign.shape, y_test_benign.shape)
    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=1, shuffle=True)
    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    X_train = np.vstack([X_train_benign, X_train])
    y_train = np.hstack([y_train_benign, y_train])
    X_test = np.vstack([X_test_benign, X_test])
    y_test = np.hstack([y_test_benign, y_test])
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def load_data_test(choose_folder, test_index, gafgyt=True, benign_included=True):
    X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=gafgyt,
                                   benign_included=benign_included)
    return X_test, y_test


def process_train_partial(X_train, y_train, method='ae', num_features=None):
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("--------------Training and testing in the same device----------------")
    print(X_train.shape, y_train.shape)
    start_time = time.time()
    # print("====================================Train scaler==================================")
    print("------------Scale data-----------------")
    scaler, X_train_scaled, _ = scale_data(X_train=X_train, X_test=None)
    print(X_train_scaled.shape, y_train.shape)
    print("X_train_scaled")
    print(X_train_scaled[:10])

    # # normalize data
    print("---------Method `{}` is used for feacture extraction (it can be `ae` or `pca`)-----------".format(method))
    encoder, pca = None, None
    if method.__eq__('ae'):
        print("-------------AE process-------------")
        X_train_scaled, encoder = ae_process(X=X_train_scaled, num_features=num_features)
    if method.__eq__('dae'):
        print("-------------DAE process------------")
        X_train_scaled, encoder = denoising_ae_process(X=X_train_scaled, num_features=num_features)
    elif method.__eq__('pca'):
        print("PCA process")
        # X_train_scaled, X_test_scaled, sfm = pca_process(X=X_train_scaled)
        X_train_scaled, pca = pca_process(X=X_train_scaled, num_features=num_features)
    print(X_train_scaled.shape, y_train.shape)

    # print("=================================Normalization====================================")
    print("---------Normalize data--------------")
    # normalize data from scaled data
    X_train_normalized, _ = normalize_data(X_train=X_train_scaled, X_test=None)
    print(X_train_normalized.shape, y_train.shape)

    # train SOM
    print("--------------------Train SOM on normalized data--------------")
    som, winmap, outliers_percentage = train_som(X_train=X_train_normalized, y_train=y_train)
    end_time = time.time()
    print('Total train time: {}'.format(end_time - start_time))
    return som, winmap, outliers_percentage, scaler, encoder, pca

def process_train_som_hyperopt(X_train, y_train, algo=tpe.suggest, som_x=None, som_y=None, sigma=6, learning_rate=2.0, verbose=False, show_progressbar = False):
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("--------------Training and testing SOM in the same device----------------")
    print(X_train.shape, y_train.shape)
    start_time = time.time()
    # print("====================================Train scaler==================================")
    print("------------Scale data-----------------")
    scaler, X_train_scaled, _ = scale_data(X_train=X_train, X_test=None)
    print(X_train_scaled.shape, y_train.shape)
    print("X_train_scaled")
    print(X_train_scaled[:10])
    # # normalize data
    print(X_train_scaled.shape, y_train.shape)
    # print("=================================Normalization====================================")
    print("---------Normalize data--------------")
    # normalize data from scaled data
    X_train_normalized, _ = normalize_data(X_train=X_train_scaled, X_test=None)
    print(X_train_normalized.shape, y_train.shape)
    # train SOM
    print("--------------------Train SOM on normalized data--------------")
    som, winmap, outliers_percentage = train_som(X_train=X_train_normalized, y_train=y_train, algo=algo, som_x=som_x, som_y=som_y, sigma=sigma, learning_rate=learning_rate, verbose=verbose, show_progressbar=show_progressbar)
    end_time = time.time()
    print('Total train time: {}'.format(end_time - start_time))
    return som, winmap, outliers_percentage, scaler


def process_train_test_partial(X_train, y_train, X_test, y_test, method='ae', num_features=None):
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("--------------Training and testing in the same device----------------")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    start_time = time.time()
    # scale data from initial data
    # print("====================================Train scaler==================================")
    print("------------Scale data-----------------")
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train=X_train, X_test=X_test)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)
    print("X_train_scaled")
    print(X_train_scaled[:10])

    # # normalize data
    print("---------Method `{}` is used for feacture extraction (it can be `ae` or `tree`)-----------".format(method))
    encoder, pca = None, None
    if method.__eq__('ae'):
        print("-------------AE process-------------")
        X_train_scaled, encoder = ae_process(X=X_train_scaled, num_features=num_features)
        X_test_scaled = encoder.predict(X_test_scaled)
    # elif method.__eq__('tree'):
    #     print("Tree process")
    #     X_train_scaled, X_test_scaled, sfm = tree_process(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled)
    elif method.__eq__('pca'):
        print("PCA process")
        # X_train_scaled, X_test_scaled, sfm = pca_process(X=X_train_scaled)
        X_train_scaled, pca = pca_process(X=X_train_scaled, num_features=num_features)
        X_test_scaled = pca.transform(X_test_scaled)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)

    # print("=================================Normalization====================================")
    print("---------Normalize data--------------")
    # normalize data from scaled data
    X_train_normalized, X_test_normalized = normalize_data(X_train=X_train_scaled, X_test=X_test_scaled)
    print(X_train_normalized.shape, y_train.shape, X_test_normalized.shape, y_test.shape)

    # train SOM
    print("--------------------Train SOM on normalized data--------------")
    som, winmap, outliers_percentage = train_som(X_train=X_train_normalized, y_train=y_train)
    end_time = time.time()
    print('Total train time: {}'.format(end_time - start_time))

    # process SOM
    print("---------------------Trainging and Testing process------------------")
    start_time = time.time()
    print("Testing on data with the same type as train data")
    test_som(som_turned=som, winmap=winmap, outliers_percentage=outliers_percentage, X_test=X_test_normalized,
             y_test=y_test)
    print("--------------------------TRAINING AND TESTING PARTIAL DONE------------------------")

    return som, winmap, outliers_percentage, scaler, encoder, pca


def process_test_partial(X_test, y_test, som, winmap, outliers_percentage, scaler, encoder=None, pca=None):
    som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
             y_test=y_test, encoder=encoder, pca=pca)


def process(choose_folder, train_index, test_indexes, method='ae', train_size=1.0, test_size=1.0,
            included_benign_in_test=False, gafgyt=False):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Train and test partial in device {}".format(train_index))
    X_train, y_train, X_test, y_test = load_common_data(choose_folder=choose_folder, train_index=train_index,
                                                        train_size=train_size, test_size=test_size, gafgyt=gafgyt)
    # load test data for other attack type
    X_test, y_test = load_data_test(choose_folder=choose_folder, test_index=train_index, gafgyt=not gafgyt,
                                    benign_included=True)
    # train AE-SOM
    som_ae, winmap_ae, outliers_percentage_ae, scaler_ae, encoder_ae, sfm_ae = process_train_test_partial(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, method='ae')
    # train TREE-SOM
    som_tree, winmap_tree, outliers_percentage_tree, scaler_tree, encoder_tree, sfm_tree = process_train_test_partial(
        X_train=X_train,
        y_train=y_train, X_test=X_test,
        y_test=y_test, method='tree')
    # train normal SOM
    som, winmap, outliers_percentage, scaler, encoder, sfm = process_train_test_partial(
        X_train=X_train,
        y_train=y_train, X_test=X_test,
        y_test=y_test, method='ae')

    print("------Test on the same device with other attack type--------")

    som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
             y_test=y_test, encoder=encoder, pca=sfm)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if test_indexes is None or not isinstance(test_indexes, list) or len(test_indexes) == 0:
        print("Test indexes must be a list of numbers.")
        return
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("===============TESTING PROCESS ON OTHER DEVICES=================")
    for test_index in test_indexes:
        print("============================================================")
        print("Test on DEVICE {}".format(test_index))
        if test_index == 3:  # device 2 contains only gafgyt data
            print("-------Device {} contains only gafgyt attack--------")
            X_test, y_test = load_data_test(choose_folder=choose_folder, test_index=test_index, gafgyt=True)
            som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
                     y_test=y_test, encoder=encoder, pca=sfm)
        else:
            print("--------Test on Mirai and Benign------------")
            X_test, y_test = load_data_test(choose_folder=choose_folder, test_index=test_index, gafgyt=gafgyt, )
            print("--------Test on Gafgyt and Benign-----------")


def main(choose_folder, train_index, test_indexes, method='ae', train_size=1.0, test_size=1.0,
         included_benign_in_test=False, gafgyt=False):
    '''
    Get training and testing data from device `train_index` in chosen folder.

    Get (anomaly) testing data from devices `test_indexes`in chosen folder.

    Steps:
        1. Get training and testing data from chosen folder
        2. Get testing data for other devices in chosen folder.

    :param choose_folder: The folder to load data.
    :param train_index: The device type used to train
    :param test_indexes: The list of devices used to test
    :param method: The method used to prepare data. It can be 'som', 'ae', or 'tree'.
    :param train_size: The train_sizex to set in function train_test_split (default is 1, and only apply other values on full data).
    :param test_size: The test_size to split the test dataset
    :param included_benign_in_test: Test only attack data or both bengin and attack together.
    :param gafgyt: test for gafgyt attack type.
    '''
    print("===========================Start=====================================")
    start_time = time.time()
    if gafgyt:
        print("Get training and testing on gafgyt data")
    else:
        print("Get traing and testing on mirai data")
    # get train and test data
    X_train, y_train, X_test, y_test = get_train_test_data(choose_folder=choose_folder,
                                                           choose_index=train_index, gafgyt=gafgyt)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=1, shuffle=True)
    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        # if X_gafgyt is not None and y_gafgyt is not None: # for testing only
        #     X_gafgyt, _, y_gafgyt, _ = train_test_split(X_gafgyt, y_gafgyt, test_size=train_size, random_state=1,
        #                                                 shuffle=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # scale data from initial data
    print("====================================Train scaler==================================")
    print("Scale data")
    scaler, X_train_scaled, X_test_scaled = scale_data(X_train=X_train, X_test=X_test)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)

    # #     data = resample(data)
    # data_benign_train_scaled = scaler.transform(data_benign)
    # data_train_scaled = scaler.transform(data_train)
    # data_test_scaled = scaler.transform(data_test)
    # # for other
    # data_gafgyt_combo_train_scaled = scaler.transform(data_gafgyt_combo_train)
    # data_gafgyt_junk_train_scaled = scaler.transform(data_gafgyt_junk_train)
    # data_gafgyt_scan_train_scaled = scaler.transform(data_gafgyt_scan_train)
    # data_gafgyt_tcp_train_scaled = scaler.transform(data_gafgyt_tcp_train)
    # data_gafgyt_udp_train_scaled = scaler.transform(data_gafgyt_udp_train)
    #
    # data_mirai_ack_train_scaled = scaler.transform(data_mirai_ack_train)
    # data_mirai_scan_train_scaled = scaler.transform(data_mirai_scan_train)
    # data_mirai_syn_train_scaled = scaler.transform(data_mirai_syn_train)
    # data_mirai_udp_train_scaled = scaler.transform(data_mirai_udp_train)
    # data_mirai_udpplain_train_scaled = scaler.transform(data_mirai_udpplain_train)

    ## With SOM --> train on Benign --> test on other
    # creating and training SOM
    # n = 5000
    # x = int(1 / 2 * np.sqrt(n))
    # som_og = minisom.MiniSom(x=x,
    #                          y=x,
    #                          input_len=X_train_scaled.shape[1],
    #                          sigma=2,
    #                          learning_rate=0.5)
    # som_og.train_random(X_train_scaled, 100)
    # # plot som
    # plot_som(som_og, X_train_scaled[0:n, :], y_train[0:n])
    # plot_som(som_og, X_test_scaled[0:n, :], y_test[0:n])

    # # normalize data
    print(
        "=================================Method `{}` is used for feacture extraction (it can be `ae` or `tree`)=====================================".format(
            method))
    encoder, sfm = None, None
    if method.__eq__('ae'):
        print("AE process")
        X_train_scaled, X_test_scaled, encoder = ae_process(X=X_train_scaled, X_test=X_test_scaled)
    elif method.__eq__('tree'):
        print("Tree process")
        X_train_scaled, X_test_scaled, sfm = tree_process(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled)
    print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)

    print("=================================Normalization====================================")
    print("Normalize data")
    # normalize data from scaled data
    X_train_normalized, X_test_normalized = normalize_data(X_train=X_train_scaled, X_test=X_test_scaled)
    print(X_train_normalized.shape, y_train.shape, X_test_normalized.shape, y_test.shape)

    # train SOM
    print("======================Train SOM on normalized data================================")
    som, winmap, outliers_percentage = train_som(X_train=X_train_normalized, y_train=y_train)
    end_time = time.time()
    print('Total train time: {}'.format(end_time - start_time))

    # process SOM
    print("====================================Testing process==================================")
    start_time = time.time()
    print("Testing on data with the same type as train data")
    test_som(som_turned=som, winmap=winmap, outliers_percentage=outliers_percentage, X_test=X_test_normalized,
             y_test=y_test)
    # process_som(X_train_normalized=X_train_normalized, y_train=y_train, X_test_normalized=X_test_normalized, y_test=y_test)

    # if test_with_gafgyt on mirai is True --> test on gafgy
    print("Testing on data with different attack type to train data type but on the same device")
    # get test data
    if gafgyt:  # test on mirai
        print("Get Mirai data")
    else:  # test on gafgyt
        print("Get Gafgyt data")
    print("Benign included")
    X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=train_index, gafgyt=not gafgyt,
                                   benign_included=True)
    som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
             y_test=y_test,
             encoder=encoder, pca=sfm)
    print("Benign is not included")
    X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=train_index, gafgyt=not gafgyt,
                                   benign_included=False)
    som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
             y_test=y_test,
             encoder=encoder, pca=sfm)

    print("======================================================")
    print("------------TESTING ON OTHER DEVICES------------------")
    for test_index in test_indexes:
        print("Perform testing on device {}".format(test_index))
        # test on gafgyt
        print("Test on gafgyt")
        print("------Include Benign------")
        X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=True,
                                       benign_included=True)
        # split data
        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
                 y_test=y_test, encoder=encoder, pca=sfm)
        print("-------Does not Included Benign-----")
        X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=True,
                                       benign_included=False)
        # split data
        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
                 y_test=y_test, encoder=encoder, pca=sfm)

        # test on mirai
        print("Test on mirai")
        print("------Include Benign------")
        X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=False,
                                       benign_included=True)
        # split data
        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
                 y_test=y_test, encoder=encoder, pca=sfm)
        print("-------Does not Included Benign-----")
        X_test, y_test = get_test_data(choose_folder=choose_folder, test_index=test_index, gafgyt=False,
                                       benign_included=False)
        # split data
        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=test_size, random_state=1, shuffle=True)
        som_test(som=som, winmap=winmap, outliers_percentage=outliers_percentage, scaler=scaler, X_test=X_test,
                 y_test=y_test, encoder=encoder, pca=sfm)
        del X_test
        del y_test
        print("-------------")
    if X_train is not None:
        del X_train
    if X_train_scaled is not None:
        del X_train_scaled
    if X_train_normalized is not None:
        del X_train_normalized
    if y_train is not None:
        del y_train
    # if X_test is not None:
    #     del X_test
    if X_test_scaled is not None:
        del X_test_scaled
    if X_test_normalized is not None:
        del X_test_normalized
    gc.collect()
    end_time = time.time()
    print('Total test time: {}'.format(end_time - start_time))
    print("======================================================================")
    # print(end_time - start_time)
    # ## Anomalies detection ##
    # training = data_benign_new
    # evaluation = data_new
    #
    # # initialize our anomaly detector with some arbitrary parameters
    # anomaly_detector = AnomalyDetection(shape=(10, 10),
    #                                     input_size=training.shape[1],
    #                                     learning_rate=8,
    #                                     learning_decay=0.001,
    #                                     initial_radius=2,
    #                                     radius_decay=0.001,
    #                                     min_number_per_bmu=0,
    #                                     number_of_neighbors=3)
    # # fit the anomaly detector and apply to the evaluation data
    # anomaly_detector.fit(training, 5000)
    # # set limits
    # benign_metrics = anomaly_detector.evaluate(data_benign_new)
    # alpha = 3
    #
    # sd_benign = np.std(benign_metrics)
    # mean_benign = np.mean(benign_metrics)
    # lim_benign = mean_benign + alpha * sd_benign
    # pct_benign = np.percentile(benign_metrics, 99.7)
    # print("sd_benign: {}, mean_benign: {}, lim_benign: {}, pct_benign: {}".format(sd_benign, mean_benign, lim_benign,
    #                                                                               pct_benign))
    # # Visualize benign metric
    # visualizing_metric(metrics=benign_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualize all anomlies
    # anomaly_metrics = anomaly_detector.evaluate(evaluation)
    # visualizing_metric(metrics=anomaly_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing ack attack
    # ack_metrics = anomaly_detector.evaluate(data_ack_new)
    # visualizing_metric(metrics=ack_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing scan attack
    # scan_metrics = anomaly_detector.evaluate(data_scan_new)
    # visualizing_metric(metrics=scan_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing syn attack
    # syn_metrics = anomaly_detector.evaluate(data_syn_new)
    # visualizing_metric(metrics=syn_metrics, lim=lim_benign, pct=pct_benign)
    # # Visualizing udp attack
    # udp_metrics = anomaly_detector.evaluate(data_udp_new)
    # visualizing_metric(metrics=udp_metrics, lim=lim_benign, pct=pct_benign)
    #
    # # Anomaly detection
    # print("Anomaly detection")
    # metrics = [benign_metrics, anomaly_metrics, ack_metrics, scan_metrics, syn_metrics, udp_metrics]
    # metric_names = ['benign', 'all alnomalies', 'ack', 'scan', 'syn', 'udp']
    # alpha = 3
    # for metric, name in zip(metrics, metric_names):
    #     print(name)
    #     get_anomalies(benign_metrics, metric, alpha, False)


if __name__ == '__main__':
    # # test mirai
    # main_v01(choose_folder=nbaiot_1K_data_path, choose_index=1, method='ae', test_gafgyt_on_mirai=False)
    # main_v01(choose_folder=nbaiot_1K_data_path, choose_index=2, method='ae', test_gafgyt_on_mirai=False)
    # main_v01(choose_folder=nbaiot_1K_data_path, choose_index=4, method='ae', test_gafgyt_on_mirai=False)

    # train on device 1, test on device 2,4
    main(choose_folder=nbaiot_1K_data_path, train_index=1, test_indexes=[2, 4], included_benign_in_test=True,
         gafgyt=False)

    # test gafgyt
    # main(choose_folder=nbaiot_1K_data_path, choose_index=1,method='ae', gafgyt=True)
    # main(choose_folder=nbaiot_data_path, choose_index=1)
