# get all file in folders
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, normalize, StandardScaler

from utils.config import nbaiot_1K_data_path, nbaiot_10K_data_path, columns, DataType, Labels


def get_all_files(folder):
    all_files = []
    for _, _, files in os.walk(folder):
        for file in files:
            all_files.append(file)
    return sorted(all_files)

def get_data_by_type(choosen_folder):
    '''
    Load data by type of device.

    Return:

    df_bengin: The dataframe of benign data.

    df_mirai: The dataframe of mirai data.

    df_gafgyt: The dataframe of gafgyt data.
    '''

    all_files = get_all_files(folder=choosen_folder)
    benign_files = [file for file in all_files if 'benign' in file]
    # print(len(benign_files))

    mirai_files = [file for file in all_files if 'mirai' in file]
    # print(len(mirai_files))

    gafgyt_files = [file for file in all_files if 'gafgyt' in file]

    # load benign
    df_benign = pd.concat(pd.read_csv(os.path.join(choosen_folder, filename)) for filename in benign_files)

    # load mirai
    df_mirai = pd.concat(pd.read_csv(os.path.join(choosen_folder, filename)) for filename in mirai_files)

    # load gafgyt
    df_gafgyt = pd.concat(pd.read_csv(os.path.join(choosen_folder, filename)) for filename in gafgyt_files)
    return df_benign, df_mirai, df_gafgyt


def get_data_by_device_type(index, choose_folder=nbaiot_1K_data_path):
    all_files = get_all_files(folder=choose_folder)
    device_file = [file for file in all_files if str(index) in file]
    df_benign = pd.DataFrame()
    df_gafgyt_combo = pd.DataFrame()
    df_gafgyt_junk = pd.DataFrame()
    df_gafgyt_scan = pd.DataFrame()
    df_gafgyt_tcp = pd.DataFrame()
    df_gafgyt_udp = pd.DataFrame()
    df_mirai_ack = pd.DataFrame()
    df_mirai_scan = pd.DataFrame()
    df_mirai_syn = pd.DataFrame()
    df_mirai_udp = pd.DataFrame()
    df_mirai_udpplain = pd.DataFrame()
    whole_data = dict()
    for filename in device_file:
        df_c = pd.read_csv(os.path.join(choose_folder, filename), usecols=columns)
        rows = df_c.shape[0]
        # we load both mirai and gafgy
        if 'benign' in filename:
            df_benign = pd.concat([df_benign.iloc[:, :].reset_index(drop=True),
                                   df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
            whole_data[DataType.benign] = df_benign
        if 'gafgyt' in filename:
            if 'combo' in filename:
                df_gafgyt_combo = pd.concat([df_gafgyt_combo.iloc[:, :].reset_index(drop=True),
                                             df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.gafgyt_combo] = df_gafgyt_combo
            if 'junk' in filename:
                df_gafgyt_junk = pd.concat([df_gafgyt_junk.iloc[:, :].reset_index(drop=True),
                                            df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.gafgyt_junk] = df_gafgyt_junk
            if 'scan' in filename:
                df_gafgyt_scan = pd.concat([df_gafgyt_scan.iloc[:, :].reset_index(drop=True),
                                            df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data['gafgyt_scan'] = df_gafgyt_scan
            if 'tcp' in filename:
                df_gafgyt_tcp = pd.concat([df_gafgyt_tcp.iloc[:, :].reset_index(drop=True),
                                           df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.gafgyt_tcp] = df_gafgyt_tcp
            if 'udp' in filename:
                df_gafgyt_udp = pd.concat([df_gafgyt_udp.iloc[:, :].reset_index(drop=True),
                                           df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.gafgyt_udp] = df_gafgyt_udp
        if 'mirai' in filename:
            if 'ack' in filename:
                df_mirai_ack = pd.concat([df_mirai_ack.iloc[:, :].reset_index(drop=True),
                                          df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.mirai_ack] = df_mirai_ack
            if 'scan' in filename:
                df_mirai_scan = pd.concat([df_mirai_scan.iloc[:, :].reset_index(drop=True),
                                           df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.mirai_scan] = df_mirai_scan
            if 'syn' in filename:
                df_mirai_syn = pd.concat([df_mirai_syn.iloc[:, :].reset_index(drop=True),
                                          df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.mirai_syn] = df_mirai_syn
            if 'udp' in filename and not 'udpplain' in filename:
                df_mirai_udp = pd.concat([df_mirai_udp.iloc[:, :].reset_index(drop=True),
                                          df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.mirai_udp] = df_mirai_udp
            if 'udpplain' in filename:
                df_mirai_udpplain = pd.concat([df_mirai_udpplain.iloc[:, :].reset_index(drop=True),
                                               df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
                whole_data[DataType.mirai_udpplain] = df_mirai_udpplain

    return whole_data  # df_benign, df_mirai_ack, df_mirai_scan, df_mirai_syn, df_mirai_udp,


#     print(device_file)

def get_data_by_device_index(choosen_folder, index):
    '''
    Get data by index (index of device in list)
    '''
    if index not in range(1, 10):
        raise Exception("Index must be in range [1,9]")
    all_files = get_all_files(folder=choosen_folder)
    device_file = [file for file in all_files if str(index) in file]
    #     print(files)
    df_benign = pd.DataFrame()
    df = pd.DataFrame()
    y_benign = []
    y = []

    for i in range(len(device_file)):
        fname = str(device_file[i])
        df_c = pd.read_csv(os.path.join(nbaiot_1K_data_path, fname))
        rows = df_c.shape[0]
        print("processing", fname, "rows =", rows)
        if 'benign' in fname:
            df_benign = df_c.iloc[:rows, :].reset_index(drop=True)
            y_benign = np.ones(rows)
        else:
            y_np = np.zeros(rows)  # np.ones(rows) if 'benign' in fname else np.zeros(rows)
            y.extend(y_np.tolist())
            df = pd.concat([df.iloc[:, :].reset_index(drop=True),
                            df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
    X_benign = df_benign.iloc[:, :].values
    y_benign = np.array(y_benign)
    X_attack = df.iloc[:, :].values
    y_attack = np.array(y)
    df_full = pd.concat([df_benign, df], axis=0)
    X_full = df_full.iloc[:, :].values
    y_full = y_benign.extend(y.tolist())
    y_full = np.array(y_full)
    #     X = df.iloc[:,:].values
    #     y = np.array(y)
    return (X_benign, y_benign, X_attack, y_attack, X_full, y_full)


##--------------------------------------------------------------##
##--DEFINE THE FUNCTION TO PRE-DIVIDE DATA AND INCREASE DATA----##
def get_data_split_train_val_test(choosen_folder, index, validation=False):
    '''
    Get the data and split by train, validation and testing set on all device types.
    '''
    if index not in range(1, 10):
        raise Exception("Index must be in range [1,9]")
    all_files = get_all_files(folder=choosen_folder)
    device_file = [file for file in all_files if str(index) in file]
    #     print(files)
    df_full = pd.DataFrame()
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    #     df_test = pd.DataFrame()
    y_full = []
    y_train = []
    y_val = []
    #     y_test = []

    for i in range(len(device_file)):
        fname = str(device_file[i])
        # get the dataframe
        # Increase the benign data from 1000 to 5000
        if 'benign' in fname:
            df_c = pd.read_csv(os.path.join(nbaiot_10K_data_path, fname), usecols=columns)
        else:
            df_c = pd.read_csv(os.path.join(nbaiot_1K_data_path, fname), usecols=columns)
        #         df_c = pd.read_csv(os.path.join(nbaiot_data_path, fname), usecols=columns)
        rows = df_c.shape[0]
        print("processing", fname, "rows =", rows)
        # get the list of labels
        y_np = np.ones(rows) if 'benign' in fname else np.zeros(rows)

        #         # lay ra 10% du lieu
        #         df_c, _, y_np, _ = train_test_split(df_c, y_np, test_size=0.9, random_state=1)

        # extend the y_full
        y_full.extend(y_np.tolist())
        # extend the df_full
        df_full = pd.concat([df_full.iloc[:, :].reset_index(drop=True),
                             df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)

        #         # split train, val, test
        #         df_c_train, df_c_test, y_np_train, y_np_test = train_test_split(df_c, y_np,
        #                                                                         test_size=0.2, random_state=1)
        #         # extend the df_test
        #         df_test = pd.concat([df_test.iloc[:,:].reset_index(drop=True),
        #                       df_c_test.iloc[:rows,:].reset_index(drop=True)], axis=0)
        #         # extend the y_test
        #         y_test.extend(y_np_test)

        # extend the df_train
        df_train = pd.concat([df_train.iloc[:, :].reset_index(drop=True),
                              df_c.iloc[:rows, :].reset_index(drop=True)], axis=0)
        # extend the y_train
        y_train.extend(y_np)
        if validation:
            df_c_train, df_c_val, y_np_train, y_np_val = train_test_split(df_c,
                                                                          y_np, test_size=0.1,
                                                                          random_state=1)
            # extend the df_val
            df_val = pd.concat([df_val.iloc[:, :].reset_index(drop=True),
                                df_c_val.iloc[:rows, :].reset_index(drop=True)], axis=0)
            # extend the y_val
            y_val.extend(y_np_val)

    return (df_full, y_full), (df_train, y_train), (df_val, y_val)  # , (df_test, y_test)


def get_data_d3(choose_folder, choose_index):
    # Load data
    # return data_benign, target_benign, data_gafgyt, target_gafgyt, data_mirai, target_mirai
    whole_data = get_data_by_device_type(index=choose_index, choose_folder=choose_folder)
    data_benign = whole_data[DataType.benign]
    data_gafgyt_combo = whole_data[DataType.gafgyt_combo]
    data_gafgyt_junk = whole_data[DataType.gafgyt_junk]
    data_gafgyt_scan = whole_data[DataType.gafgyt_scan]
    data_gafgyt_tcp = whole_data[DataType.gafgyt_tcp]
    data_gafgyt_udp = whole_data[DataType.gafgyt_udp]

    # add column class
    data_benign['class'] = Labels[DataType.benign]
    data_gafgyt_combo['class'] = Labels[DataType.gafgyt_combo]
    data_gafgyt_junk['class'] = Labels[DataType.gafgyt_junk]
    data_gafgyt_scan['class'] = Labels[DataType.gafgyt_scan]
    data_gafgyt_tcp['class'] = Labels[DataType.gafgyt_tcp]
    data_gafgyt_udp['class'] = Labels[DataType.gafgyt_udp]

    # get labels
    target_benign = data_benign.iloc[:, -1].astype(int)
    target_gafgyt_combo = data_gafgyt_combo.iloc[:, -1].astype(int)
    target_gafgyt_junk = data_gafgyt_junk.iloc[:, -1].astype(int)
    target_gafgyt_scan = data_gafgyt_scan.iloc[:, -1].astype(int)
    target_gafgyt_tcp = data_gafgyt_tcp.iloc[:, -1].astype(int)
    target_gafgyt_udp = data_gafgyt_udp.iloc[:, -1].astype(int)

    # Get data
    data_benign = data_benign.iloc[:, 0:-1]
    data_gafgyt_combo = data_gafgyt_combo.iloc[:, 0:-1]
    data_gafgyt_junk = data_gafgyt_junk.iloc[:, 0:-1]
    data_gafgyt_scan = data_gafgyt_scan.iloc[:, 0:-1]
    data_gafgyt_tcp = data_gafgyt_tcp.iloc[:, 0:-1]
    data_gafgyt_udp = data_gafgyt_udp.iloc[:, 0:-1]

    # combine data
    data_gafgyt = np.vstack(
        [data_gafgyt_combo, data_gafgyt_junk, data_gafgyt_scan,
         data_gafgyt_tcp, data_gafgyt_udp])
    target_gafgyt = np.hstack(
        [target_gafgyt_combo, target_gafgyt_junk, target_gafgyt_scan,
         target_gafgyt_tcp, target_gafgyt_udp])

    return data_benign, target_benign, data_gafgyt, target_gafgyt


def get_data(choose_folder, choose_index):
    # Load data
    # return data_benign, target_benign, data_gafgyt, target_gafgyt, data_mirai, target_mirai
    whole_data = get_data_by_device_type(index=choose_index, choose_folder=choose_folder)
    data_benign = whole_data[DataType.benign]
    data_gafgyt_combo = whole_data[DataType.gafgyt_combo]
    data_gafgyt_junk = whole_data[DataType.gafgyt_junk]
    data_gafgyt_scan = whole_data[DataType.gafgyt_scan]
    data_gafgyt_tcp = whole_data[DataType.gafgyt_tcp]
    data_gafgyt_udp = whole_data[DataType.gafgyt_udp]

    data_mirai_ack = whole_data[DataType.mirai_ack]
    data_mirai_scan = whole_data[DataType.mirai_scan]
    data_mirai_syn = whole_data[DataType.mirai_syn]
    data_mirai_udp = whole_data[DataType.mirai_udp]
    data_mirai_udpplain = whole_data[DataType.mirai_udpplain]

    # add column class
    data_benign['class'] = Labels[DataType.benign]

    data_gafgyt_combo['class'] = Labels[DataType.gafgyt_combo]
    data_gafgyt_junk['class'] = Labels[DataType.gafgyt_junk]
    data_gafgyt_scan['class'] = Labels[DataType.gafgyt_scan]
    data_gafgyt_tcp['class'] = Labels[DataType.gafgyt_tcp]
    data_gafgyt_udp['class'] = Labels[DataType.gafgyt_udp]

    data_mirai_ack['class'] = Labels[DataType.mirai_ack]
    data_mirai_scan['class'] = Labels[DataType.mirai_scan]
    data_mirai_syn['class'] = Labels[DataType.mirai_syn]
    data_mirai_udp['class'] = Labels[DataType.mirai_udp]
    data_mirai_udpplain['class'] = Labels[DataType.mirai_udpplain]

    # get labels
    target_benign = data_benign.iloc[:, -1].astype(int)
    target_gafgyt_combo = data_gafgyt_combo.iloc[:, -1].astype(int)
    target_gafgyt_junk = data_gafgyt_junk.iloc[:, -1].astype(int)
    target_gafgyt_scan = data_gafgyt_scan.iloc[:, -1].astype(int)
    target_gafgyt_tcp = data_gafgyt_tcp.iloc[:, -1].astype(int)
    target_gafgyt_udp = data_gafgyt_udp.iloc[:, -1].astype(int)

    target_mirai_ack = data_mirai_ack.iloc[:, -1].astype(int)
    target_mirai_scan = data_mirai_scan.iloc[:, -1].astype(int)
    target_mirai_syn = data_mirai_syn.iloc[:, -1].astype(int)
    target_mirai_udp = data_mirai_udp.iloc[:, -1].astype(int)
    target_mirai_udpplain = data_mirai_udpplain.iloc[:, -1].astype(int)

    # Get data
    data_benign = data_benign.iloc[:, 0:-1]
    data_gafgyt_combo = data_gafgyt_combo.iloc[:, 0:-1]
    data_gafgyt_junk = data_gafgyt_junk.iloc[:, 0:-1]
    data_gafgyt_scan = data_gafgyt_scan.iloc[:, 0:-1]
    data_gafgyt_tcp = data_gafgyt_tcp.iloc[:, 0:-1]
    data_gafgyt_udp = data_gafgyt_udp.iloc[:, 0:-1]
    data_mirai_ack = data_mirai_ack.iloc[:, 0:-1]
    data_mirai_scan = data_mirai_scan.iloc[:, 0:-1]
    data_mirai_syn = data_mirai_syn.iloc[:, 0:-1]
    data_mirai_udp = data_mirai_udp.iloc[:, 0:-1]
    data_mirai_udpplain = data_mirai_udpplain.iloc[:, 0:-1]

    # combine data
    data_gafgyt = np.vstack(
        [data_gafgyt_combo, data_gafgyt_junk, data_gafgyt_scan,
         data_gafgyt_tcp, data_gafgyt_udp])
    target_gafgyt = np.hstack(
        [target_gafgyt_combo, target_gafgyt_junk, target_gafgyt_scan,
         target_gafgyt_tcp, target_gafgyt_udp])

    data_mirai = np.vstack([data_mirai_ack, data_mirai_scan,
                            data_mirai_syn, data_mirai_udp, data_mirai_udpplain])
    target_mirai = np.hstack(
        [target_mirai_ack, target_mirai_scan, target_mirai_syn,
         target_mirai_udp, target_mirai_udpplain])

    return data_benign, target_benign, data_gafgyt, target_gafgyt, data_mirai, target_mirai


def get_train_test_data(choose_folder, choose_index, gafgyt=False):
    # Load data
    whole_data = get_data_by_device_type(index=choose_index, choose_folder=choose_folder)
    data_benign = whole_data[DataType.benign]
    data_gafgyt_combo = whole_data[DataType.gafgyt_combo]
    data_gafgyt_junk = whole_data[DataType.gafgyt_junk]
    data_gafgyt_scan = whole_data[DataType.gafgyt_scan]
    data_gafgyt_tcp = whole_data[DataType.gafgyt_tcp]
    data_gafgyt_udp = whole_data[DataType.gafgyt_udp]
    try:
        data_mirai_ack = whole_data[DataType.mirai_ack]
        data_mirai_scan = whole_data[DataType.mirai_scan]
        data_mirai_syn = whole_data[DataType.mirai_syn]
        data_mirai_udp = whole_data[DataType.mirai_udp]
        data_mirai_udpplain = whole_data[DataType.mirai_udpplain]
    except Exception as e:
        print("Perhaps they are device 3 or 7")
        print(e.__str__())

    # add column class
    data_benign['class'] = Labels[DataType.benign]

    data_gafgyt_combo['class'] = Labels[DataType.gafgyt_combo]
    data_gafgyt_junk['class'] = Labels[DataType.gafgyt_junk]
    data_gafgyt_scan['class'] = Labels[DataType.gafgyt_scan]
    data_gafgyt_tcp['class'] = Labels[DataType.gafgyt_tcp]
    data_gafgyt_udp['class'] = Labels[DataType.gafgyt_udp]
    try:
        data_mirai_ack['class'] = Labels[DataType.mirai_ack]
        data_mirai_scan['class'] = Labels[DataType.mirai_scan]
        data_mirai_syn['class'] = Labels[DataType.mirai_syn]
        data_mirai_udp['class'] = Labels[DataType.mirai_udp]
        data_mirai_udpplain['class'] = Labels[DataType.mirai_udpplain]
    except Exception as e:
        print(e.__str__())

    # get labels
    target_benign = data_benign.iloc[:, -1].astype(int)
    target_gafgyt_combo = data_gafgyt_combo.iloc[:, -1].astype(int)
    target_gafgyt_junk = data_gafgyt_junk.iloc[:, -1].astype(int)
    target_gafgyt_scan = data_gafgyt_scan.iloc[:, -1].astype(int)
    target_gafgyt_tcp = data_gafgyt_tcp.iloc[:, -1].astype(int)
    target_gafgyt_udp = data_gafgyt_udp.iloc[:, -1].astype(int)
    try:
        target_mirai_ack = data_mirai_ack.iloc[:, -1].astype(int)
        target_mirai_scan = data_mirai_scan.iloc[:, -1].astype(int)
        target_mirai_syn = data_mirai_syn.iloc[:, -1].astype(int)
        target_mirai_udp = data_mirai_udp.iloc[:, -1].astype(int)
        target_mirai_udpplain = data_mirai_udpplain.iloc[:, -1].astype(int)
    except Exception as e:
        print(e.__str__())

    # Get data
    data_benign = data_benign.iloc[:, 0:-1]
    data_gafgyt_combo = data_gafgyt_combo.iloc[:, 0:-1]
    data_gafgyt_junk = data_gafgyt_junk.iloc[:, 0:-1]
    data_gafgyt_scan = data_gafgyt_scan.iloc[:, 0:-1]
    data_gafgyt_tcp = data_gafgyt_tcp.iloc[:, 0:-1]
    data_gafgyt_udp = data_gafgyt_udp.iloc[:, 0:-1]
    try:
        data_mirai_ack = data_mirai_ack.iloc[:, 0:-1]
        data_mirai_scan = data_mirai_scan.iloc[:, 0:-1]
        data_mirai_syn = data_mirai_syn.iloc[:, 0:-1]
        data_mirai_udp = data_mirai_udp.iloc[:, 0:-1]
        data_mirai_udpplain = data_mirai_udpplain.iloc[:, 0:-1]
    except Exception as e:
        print(e.__str__())

    # split benign for training and testing
    data_benign_train, data_benign_test, target_benign_train, target_benign_test = train_test_split(data_benign,
                                                                                                    target_benign,
                                                                                                    test_size=0.3,
                                                                                                    shuffle=True,
                                                                                                    random_state=1)

    # combine data
    if gafgyt:
        # Split data
        data_gafgyt_combo_train, data_gafgyt_combo_test, target_gafgyt_combo_train, target_gafgyt_combo_test = train_test_split(
            data_gafgyt_combo, target_gafgyt_combo, test_size=0.3, shuffle=True, random_state=1)
        data_gafgyt_junk_train, data_gafgyt_junk_test, target_gafgyt_junk_train, target_gafgyt_junk_test = train_test_split(
            data_gafgyt_junk, target_gafgyt_junk, test_size=0.3, shuffle=True, random_state=1)
        data_gafgyt_scan_train, data_gafgyt_scan_test, target_gafgyt_scan_train, target_gafgyt_scan_test = train_test_split(
            data_gafgyt_scan, target_gafgyt_scan, test_size=0.3, shuffle=True, random_state=1)
        data_gafgyt_tcp_train, data_gafgyt_tcp_test, target_gafgyt_tcp_train, target_gafgyt_tcp_test = train_test_split(
            data_gafgyt_tcp, target_gafgyt_tcp, test_size=0.3, shuffle=True, random_state=1)
        data_gafgyt_udp_train, data_gafgyt_udp_test, target_gafgyt_udp_train, target_gafgyt_udp_test = train_test_split(
            data_gafgyt_udp, target_gafgyt_udp, test_size=0.3, shuffle=True, random_state=1)

        # combine data
        # data_train = np.vstack(
        #     [data_benign_train, data_gafgyt_combo_train, data_gafgyt_junk_train, data_gafgyt_scan_train,
        #      data_gafgyt_tcp_train, data_gafgyt_udp_train])
        # target_train = np.hstack(
        #     [target_benign_train, target_gafgyt_combo_train, target_gafgyt_junk_train, target_gafgyt_scan_train,
        #      target_gafgyt_tcp_train, target_gafgyt_udp_train])

        data_train = np.vstack(
            [data_gafgyt_combo_train, data_gafgyt_junk_train, data_gafgyt_scan_train,
             data_gafgyt_tcp_train, data_gafgyt_udp_train])
        target_train = np.hstack(
            [target_gafgyt_combo_train, target_gafgyt_junk_train, target_gafgyt_scan_train,
             target_gafgyt_tcp_train, target_gafgyt_udp_train])

        # data_test = np.vstack([data_benign_test, data_gafgyt_combo_test, data_gafgyt_junk_test, data_gafgyt_scan_test,
        #                        data_gafgyt_tcp_test, data_gafgyt_udp_test])
        # target_test = np.hstack(
        #     [target_benign_test, target_gafgyt_combo_test, target_gafgyt_junk_test, target_gafgyt_scan_test,
        #      target_gafgyt_tcp_test, target_gafgyt_udp_test])

        data_test = np.vstack([data_gafgyt_combo_test, data_gafgyt_junk_test, data_gafgyt_scan_test,
                               data_gafgyt_tcp_test, data_gafgyt_udp_test])
        target_test = np.hstack(
            [target_gafgyt_combo_test, target_gafgyt_junk_test, target_gafgyt_scan_test,
             target_gafgyt_tcp_test, target_gafgyt_udp_test])

        return data_train, target_train, data_test, target_test, data_benign_train, target_benign_train, data_benign_test, target_benign_test  # , None, None

    else:
        # Split mirai data
        data_mirai_ack_train, data_mirai_ack_test, target_mirai_ack_train, target_mirai_ack_test = train_test_split(
            data_mirai_ack, target_mirai_ack, test_size=0.3, shuffle=True, random_state=1)
        data_mirai_scan_train, data_mirai_scan_test, target_mirai_scan_train, target_mirai_scan_test = train_test_split(
            data_mirai_scan, target_mirai_scan, test_size=0.3, shuffle=True, random_state=1)
        data_mirai_syn_train, data_mirai_syn_test, target_mirai_syn_train, target_mirai_syn_test = train_test_split(
            data_mirai_syn, target_mirai_syn, test_size=0.3, shuffle=True, random_state=1)
        data_mirai_udp_train, data_mirai_udp_test, target_mirai_udp_train, target_mirai_udp_test = train_test_split(
            data_mirai_udp, target_mirai_udp, test_size=0.3, shuffle=True, random_state=1)
        data_mirai_udpplain_train, data_mirai_udpplain_test, target_mirai_udpplain_train, target_mirai_udpplain_test = train_test_split(
            data_mirai_udpplain, target_mirai_udpplain, test_size=0.3, shuffle=True, random_state=1)

        # combine to make train and test data
        # data_train = np.vstack([data_benign_train, data_mirai_ack_train, data_mirai_scan_train,
        #                         data_mirai_syn_train, data_mirai_udp_train, data_mirai_udpplain_train])
        # target_train = np.hstack(
        #     [target_benign_train, target_mirai_ack_train, target_mirai_scan_train, target_mirai_syn_train,
        #      target_mirai_udp_train, target_mirai_udpplain_train])
        data_train = np.vstack([data_mirai_ack_train, data_mirai_scan_train,
                                data_mirai_syn_train, data_mirai_udp_train, data_mirai_udpplain_train])
        target_train = np.hstack(
            [target_mirai_ack_train, target_mirai_scan_train, target_mirai_syn_train,
             target_mirai_udp_train, target_mirai_udpplain_train])

        # data_test = np.vstack(
        #     [data_benign_test, data_mirai_ack_test, data_mirai_scan_test, data_mirai_syn_test, data_mirai_udp_test,
        #      data_mirai_udpplain_test])
        # target_test = np.hstack(
        #     [target_benign_test, target_mirai_ack_test, target_mirai_scan_test, target_mirai_syn_test,
        #      target_mirai_udp_test, target_mirai_udpplain_test])
        data_test = np.vstack(
            [data_mirai_ack_test, data_mirai_scan_test, data_mirai_syn_test, data_mirai_udp_test,
             data_mirai_udpplain_test])
        target_test = np.hstack(
            [target_mirai_ack_test, target_mirai_scan_test, target_mirai_syn_test,
             target_mirai_udp_test, target_mirai_udpplain_test])

        # data_gafgyt = np.vstack(
        #     [data_gafgyt_combo, data_gafgyt_junk, data_gafgyt_scan, data_gafgyt_tcp, data_gafgyt_udp])
        # target_gafgyt = np.hstack(
        #     [target_gafgyt_combo, target_gafgyt_junk, target_gafgyt_scan, target_gafgyt_tcp, target_gafgyt_udp])
        return data_train, target_train, data_test, target_test, data_benign_train, target_benign_train, data_benign_test, target_benign_test  # , data_gafgyt, target_gafgyt


def get_test_data(choose_folder, test_index, gafgyt=True, benign_included=False):
    '''
    - Scenario 1: Keep benign
    '''
    if gafgyt:
        print("Load Gafgty data")
    else:
        print("Load Mirai data")
    if benign_included:
        print("Benign data are included in testing data")
    else:
        print("Benign data are not included in testing data")
    # Load data
    whole_data = get_data_by_device_type(index=test_index, choose_folder=choose_folder)
    data_benign = whole_data[DataType.benign]
    data_gafgyt_combo = whole_data[DataType.gafgyt_combo]
    data_gafgyt_junk = whole_data[DataType.gafgyt_junk]
    data_gafgyt_scan = whole_data[DataType.gafgyt_scan]
    data_gafgyt_tcp = whole_data[DataType.gafgyt_tcp]
    data_gafgyt_udp = whole_data[DataType.gafgyt_udp]
    if test_index not in [3, 7]:
        data_mirai_ack = whole_data[DataType.mirai_ack]
        data_mirai_scan = whole_data[DataType.mirai_scan]
        data_mirai_syn = whole_data[DataType.mirai_syn]
        data_mirai_udp = whole_data[DataType.mirai_udp]
        data_mirai_udpplain = whole_data[DataType.mirai_udpplain]

    # add column class
    data_benign['class'] = Labels[DataType.benign]

    data_gafgyt_combo['class'] = Labels[DataType.gafgyt_combo]
    data_gafgyt_junk['class'] = Labels[DataType.gafgyt_junk]
    data_gafgyt_scan['class'] = Labels[DataType.gafgyt_scan]
    data_gafgyt_tcp['class'] = Labels[DataType.gafgyt_tcp]
    data_gafgyt_udp['class'] = Labels[DataType.gafgyt_udp]
    if test_index not in [3, 7]:
        data_mirai_ack['class'] = Labels[DataType.mirai_ack]
        data_mirai_scan['class'] = Labels[DataType.mirai_scan]
        data_mirai_syn['class'] = Labels[DataType.mirai_syn]
        data_mirai_udp['class'] = Labels[DataType.mirai_udp]
        data_mirai_udpplain['class'] = Labels[DataType.mirai_udpplain]

    # get labels
    target_benign = data_benign.iloc[:, -1].astype(int)
    target_gafgyt_combo = data_gafgyt_combo.iloc[:, -1].astype(int)
    target_gafgyt_junk = data_gafgyt_junk.iloc[:, -1].astype(int)
    target_gafgyt_scan = data_gafgyt_scan.iloc[:, -1].astype(int)
    target_gafgyt_tcp = data_gafgyt_tcp.iloc[:, -1].astype(int)
    target_gafgyt_udp = data_gafgyt_udp.iloc[:, -1].astype(int)
    if test_index not in [3, 7]:
        target_mirai_ack = data_mirai_ack.iloc[:, -1].astype(int)
        target_mirai_scan = data_mirai_scan.iloc[:, -1].astype(int)
        target_mirai_syn = data_mirai_syn.iloc[:, -1].astype(int)
        target_mirai_udp = data_mirai_udp.iloc[:, -1].astype(int)
        target_mirai_udpplain = data_mirai_udpplain.iloc[:, -1].astype(int)

    # Get data
    data_benign = data_benign.iloc[:, 0:-1]
    data_gafgyt_combo = data_gafgyt_combo.iloc[:, 0:-1]
    data_gafgyt_junk = data_gafgyt_junk.iloc[:, 0:-1]
    data_gafgyt_scan = data_gafgyt_scan.iloc[:, 0:-1]
    data_gafgyt_tcp = data_gafgyt_tcp.iloc[:, 0:-1]
    data_gafgyt_udp = data_gafgyt_udp.iloc[:, 0:-1]
    if test_index not in [3, 7]:
        data_mirai_ack = data_mirai_ack.iloc[:, 0:-1]
        data_mirai_scan = data_mirai_scan.iloc[:, 0:-1]
        data_mirai_syn = data_mirai_syn.iloc[:, 0:-1]
        data_mirai_udp = data_mirai_udp.iloc[:, 0:-1]
        data_mirai_udpplain = data_mirai_udpplain.iloc[:, 0:-1]

    # return data without splitting
    # combine data
    if gafgyt:
        # combine data
        data = np.vstack([data_gafgyt_combo, data_gafgyt_junk, data_gafgyt_scan, data_gafgyt_tcp, data_gafgyt_udp])
        target = np.hstack(
            [target_gafgyt_combo, target_gafgyt_junk, target_gafgyt_scan, target_gafgyt_tcp, target_gafgyt_udp])
        # if len(target) > 400000:
        #     test_size = 0.7
        # else:
        #     test_size = 0.5
        # data, _, target, _ = train_test_split(data, target, test_size=test_size, shuffle=True, random_state=1)
    else:
        if test_index not in [3, 7]:
            # combine to make train and test data
            data = np.vstack([data_mirai_ack, data_mirai_scan, data_mirai_syn, data_mirai_udp, data_mirai_udpplain])
            target = np.hstack(
                [target_mirai_ack, target_mirai_scan, target_mirai_syn, target_mirai_udp, target_mirai_udpplain])
            # if len(target) > 400000:
            #     test_size = 0.7
            # else:
            #     test_size = 0.5
            # data, _, target, _ = train_test_split(data, target, test_size=test_size, shuffle=True, random_state=1)
    if benign_included:
        data = np.vstack([data_benign, data])
        target = np.hstack([target_benign, target])
    # if len(target) > 400000:
    #     test_size = 0.5
    # else:
    #     test_size = 0.2
    # data, _, target, _ = train_test_split(data, target, test_size=test_size, shuffle=True, random_state=1)
    # handle here for too large dataset
    return data, target
