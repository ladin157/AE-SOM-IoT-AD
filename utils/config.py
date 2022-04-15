import os

# data_path = '../data'
project_path = os.path.dirname(os.path.dirname(__file__))
# print(project_path)
data_path = os.path.join(project_path, 'data')
nbaiot_data_path = os.path.join(data_path, 'nbaiot')
nbaiot_20K_data_path = os.path.join(data_path, 'nbaiot_20K')
nbaiot_10K_data_path = os.path.join(data_path, 'nbaiot_10K')
nbaiot_5K_data_path = os.path.join(data_path, 'nbaiot_5K')
nbaiot_1K_data_path = os.path.join(data_path, 'nbaiot_1K')

group_devices = [
    {
        "indexes": [1, 3],
        "names": ["Danmini_Doorbell", "Ennio_Doorbell"]
    },
    {
        "indexes": [5, 6, 8, 9],
        "names": ["Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera",
                  "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"]
    }]

# Device family
dn_nbaiot = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
             'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
             'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
             'SimpleHome_XCS7_1003_WHT_Security_Camera']
# print(os.path.exists(nbaiot_1K_data_path))

# Columns
columns = ["MI_dir_L5_weight", "MI_dir_L5_mean", "MI_dir_L5_variance", "MI_dir_L3_weight", "MI_dir_L3_mean",
           "MI_dir_L3_variance", "MI_dir_L1_weight", "MI_dir_L1_mean", "MI_dir_L1_variance",
           "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance", "MI_dir_L0.01_weight",
           "MI_dir_L0.01_mean", "MI_dir_L0.01_variance", "H_L5_weight", "H_L5_mean", "H_L5_variance",
           "H_L3_weight", "H_L3_mean", "H_L3_variance", "H_L1_weight", "H_L1_mean", "H_L1_variance",
           "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance", "H_L0.01_weight", "H_L0.01_mean",
           "H_L0.01_variance", "HH_L5_weight", "HH_L5_mean", "HH_L5_std", "HH_L5_magnitude",
           "HH_L5_radius", "HH_L5_covariance", "HH_L5_pcc", "HH_L3_weight", "HH_L3_mean",
           "HH_L3_std", "HH_L3_magnitude", "HH_L3_radius", "HH_L3_covariance", "HH_L3_pcc",
           "HH_L1_weight", "HH_L1_mean", "HH_L1_std", "HH_L1_magnitude", "HH_L1_radius",
           "HH_L1_covariance", "HH_L1_pcc", "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std",
           "HH_L0.1_magnitude", "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
           "HH_L0.01_weight", "HH_L0.01_mean", "HH_L0.01_std", "HH_L0.01_magnitude",
           "HH_L0.01_radius", "HH_L0.01_covariance", "HH_L0.01_pcc", "HH_jit_L5_weight",
           "HH_jit_L5_mean", "HH_jit_L5_variance", "HH_jit_L3_weight", "HH_jit_L3_mean",
           "HH_jit_L3_variance", "HH_jit_L1_weight", "HH_jit_L1_mean", "HH_jit_L1_variance",
           "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance", "HH_jit_L0.01_weight",
           "HH_jit_L0.01_mean", "HH_jit_L0.01_variance", "HpHp_L5_weight", "HpHp_L5_mean",
           "HpHp_L5_std", "HpHp_L5_magnitude", "HpHp_L5_radius", "HpHp_L5_covariance",
           "HpHp_L5_pcc", "HpHp_L3_weight", "HpHp_L3_mean", "HpHp_L3_std", "HpHp_L3_magnitude",
           "HpHp_L3_radius", "HpHp_L3_covariance", "HpHp_L3_pcc", "HpHp_L1_weight", "HpHp_L1_mean",
           "HpHp_L1_std", "HpHp_L1_magnitude", "HpHp_L1_radius", "HpHp_L1_covariance", "HpHp_L1_pcc",
           "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std", "HpHp_L0.1_magnitude",
           "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc", "HpHp_L0.01_weight",
           "HpHp_L0.01_mean", "HpHp_L0.01_std", "HpHp_L0.01_magnitude", "HpHp_L0.01_radius",
           "HpHp_L0.01_covariance", "HpHp_L0.01_pcc"]


class DataType:
    benign = 'benign'

    gafgyt_combo = 'gafgyt_combo'
    gafgyt_junk = 'gafgyt_junk'
    gafgyt_scan = 'gafgyt_scan'
    gafgyt_tcp = 'gafgyt_tcp'
    gafgyt_udp = 'gafgyt_udp'

    mirai_ack = 'mirai_ack'
    mirai_scan = 'mirai_scan'
    mirai_syn = 'mirai_syn'
    mirai_udp = 'mirai_udp'
    mirai_udpplain = 'mirai_udpplain'


Labels = {
    'benign': 1,
    'gafgyt_combo': 2,
    'gafgyt_junk': 3,
    'gafgyt_scan': 4,
    'gafgyt_tcp': 5,
    'gafgyt_udp': 6,
    'mirai_ack': 2,
    'mirai_scan': 3,
    'mirai_syn': 4,
    'mirai_udp': 5,
    'mirai_udpplain': 6
}

markers = ['o', 's', 'D', '+', 'v', 'p', '*', 'x', '^', '<', '>', '8']
colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange', 'purple', 'grey', 'cyan', 'brown', 'olive', 'lime']
# ----------Training setting----------#
