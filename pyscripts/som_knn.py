from sklearn.model_selection import train_test_split

from pyscripts.som_common import process_som
from utils.config import nbaiot_1K_data_path
from utils.datasets import get_train_test_data
from utils.preprocessing import scale_data, normalize_data

'''
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

def main(choose_folder, choose_index, train_size=1, gafgyt=False, test_gafgyt_on_mirai=False):
    '''
    :param choose_folder: The folder to load data.
    :param choose_index: The device type to test.
    :param train_size: The train_size to set in function train_test_split (default is 1, and only apply other values on full data).
    :param gafgyt: test for gafgyt attack type.
    :param test_gafgyt_on_mirai: Train on mirai, but use gafgyt for testing.
    '''
    print("======================================================================")
    if gafgyt:
        print("Training and testing on gafgyt data")
    else:
        print("Traing and testing on mirai data")
    # get train and test data
    X_train, y_train, X_test, y_test, X_gafgyt, y_gafgyt = get_train_test_data(choose_folder=choose_folder,
                                                                               choose_index=choose_index, gafgyt=gafgyt)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=1, shuffle=True)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=train_size, random_state=1, shuffle=True)
    X_gafgyt, y_gafgyt = train_test_split(X_gafgyt, y_gafgyt, train_size=train_size, random_state=1, shuffle=True)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # scale data from initial data
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

    print("Normalize data")
    # normalize data from scaled data
    X_train_normalized, X_test_normalized = normalize_data(X_train=X_train_scaled, X_test=X_test_scaled)
    print(X_train_normalized.shape, y_train.shape, X_test_normalized.shape, y_test.shape)

    # process SOM
    process_som(X_train_normalized=X_train_normalized, y_train=y_train, X_test_normalized=X_test_normalized,
                y_test=y_test)

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
    main(choose_folder=nbaiot_1K_data_path, choose_index=1)
