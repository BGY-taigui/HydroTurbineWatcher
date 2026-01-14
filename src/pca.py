import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def read_row_data(csv_file_name):
    with open(csv_file_name, 'r') as file:
        data = file.readlines()
    loaded_data = [list(map(float, line.strip().split(','))) for line in data]
    return [
        [item[index] for item in loaded_data] for index in range(len(loaded_data[0]))
        ]

def get_feature_values(row_values):
    #算術平均　中央値　標準偏差　分散　尖度　歪度　最大　最小　範囲
    mean_values = sum(row_values) / len(row_values)

    median_values = np.median(row_values)

    std_values = np.std(row_values)

    var_values = np.var(row_values)

    skewness = (np.mean((row_values - mean_values) ** 3)) / (std_values ** 3)

    kurtosis = (np.mean((row_values - mean_values) ** 4)) / (std_values ** 4)

    max_values = np.max(row_values)
    min_values = np.min(row_values)

    range_values = max_values - min_values

    return [
        mean_values,
        median_values,
        std_values,
        var_values,
        skewness,
        kurtosis,
        max_values,
        min_values,
        range_values
    ]

def get_fft_power_band(row_values,power_band_num=20,hanning_window=False):

    #window function
    if hanning_window:
        window = np.hanning(len(row_values))
        values_windowed = row_values * window

        amps = [np.abs(item) for item in np.fft.fft(values_windowed)]
    else:
        amps = [np.abs(item) for item in np.fft.fft(row_values)]

    amp_data_num = len(amps)

    effective_amp_data_num = int(amp_data_num // 2.5)

    power_band_range = int(effective_amp_data_num // power_band_num)

    power_band = [
        sum(map(lambda x: x**2, amps[index*power_band_range:(index+1)*power_band_range])) 
        for index in range(int(effective_amp_data_num//power_band_range))
    ]

    return power_band


def get_dataset(sampling_num_for_data, row_datas,fft_window_func = False):

    trimed_row_datas = []

    for index in range(row_datas.shape[1]//sampling_num_for_data):

        trimed_row_datas.append(
            row_datas[:,index*sampling_num_for_data:(index+1)*sampling_num_for_data]
        )

    dataset = []
    for trimed_row_data in trimed_row_datas:
        dataset_for_time = []

        for row_data in trimed_row_data:
            dataset_for_time.extend(get_feature_values(row_data))

        for row_data in trimed_row_data:
            dataset_for_time.extend(get_fft_power_band(row_data,hanning_window=fft_window_func))

        dataset.append(dataset_for_time)
    return dataset

def pca(dataset):

    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    pca_model = PCA(n_components=8)
    dataset_pca = pca_model.fit_transform(dataset_scaled)

    print("Original shape:", np.array(dataset).shape)
    print("Explained variance ratio:", pca_model.explained_variance_ratio_)
    print("PCA Mtrix", pca_model.components_)


    fignum = 1
    dimnum = 1
    plt.figure(fignum) 
    for item in pca_model.components_:
        plt.plot(item,label=f"PC{dimnum}")
        dimnum += 1

    plt.legend()
    #plt.show()
    plt.savefig("PCA_test_components_GVO80_16500.png")

def pca_evalution():
    pass


if __name__ == "__main__":
    csv_file_name = "TestData/0107/GVO80-N11-61test_152328.csv"
    row_datas = np.array(read_row_data(csv_file_name))

    row_data_1dim = np.array([row_datas[0]])

    sampling_num_for_data = 16500

    dataset = get_dataset(sampling_num_for_data, row_data_1dim)

    print("Dataset Length:", len(dataset))

    pca(dataset)