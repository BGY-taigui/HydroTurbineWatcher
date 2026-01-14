import numpy as np

def get_OA_value(data,hanning_window=True):
    if hanning_window:
        window = np.hanning(len(data))
        values_windowed = data * window

        amps = [np.abs(item)*2/len(values_windowed) for item in np.fft.fft(values_windowed)][:len(data)//2]

    else:
        amps = [np.abs(item)*2/len(data) for item in np.fft.fft(data)][:len(data)//2]
    
    OA_value = sum([amps[i]**2 for i in range(len(amps))]) 

    return OA_value


def get_f_band_value(data,sampling_time,f,hanning_window=True,freq_width = 0.1):
    # サンプリング周期を計算
    dt = sampling_time / (len(data) - 1)
    
    if hanning_window:
        window = np.hanning(len(data))
        values_windowed = data * window

        amps = [np.abs(item)*2/len(values_windowed) for item in np.fft.fft(values_windowed)][:len(data)//2]
        freq = np.fft.fftfreq(len(values_windowed), d=dt)[:len(data)//2]

    else:
        amps = [np.abs(item)*2/len(data) for item in np.fft.fft(data)][:len(data)//2]
        freq = np.fft.fftfreq(len(data), d=dt)[:len(data)//2]


    freq_band_index = []
    for index in range(len(freq)):
        if freq[index] >= f * (1-freq_width) and freq[index] <= f * (1+freq_width):
            freq_band_index.append(index)

    if len(freq_band_index) == 0:
        fn_value = 0
    else:
        fn_value = sum([amps[i]**2 for i in freq_band_index])/len(freq_band_index)

    return fn_value


if __name__ == "__main__":


    filenames = [
        "TestData/0112/GVO30/GVO30-N11-46.csv",
        "TestData/0112/GVO30/GVO30-N11-48.5.csv",
        "TestData/0112/GVO30/GVO30-N11-51.csv",
        "TestData/0112/GVO30/GVO30-N11-53.5.csv",
        "TestData/0112/GVO30/GVO30-N11-56.csv",
        "TestData/0112/GVO30/GVO30-N11-58.5.csv",
        "TestData/0112/GVO30/GVO30-N11-61.csv",
        "TestData/0112/GVO30/GVO30-N11-63.5.csv",
        "TestData/0112/GVO30/GVO30-N11-66.csv",
        "TestData/0112/GVO30/GVO30-N11-68.5.csv",
        "TestData/0112/GVO30/GVO30-N11-71.csv",
        "TestData/0112/GVO30/GVO30-N11-73.5.csv",
        "TestData/0112/GVO30/GVO30-N11-76.csv",

        "TestData/0112/GVO40/GVO40-N11-46.csv",
        "TestData/0112/GVO40/GVO40-N11-48.5.csv",
        "TestData/0112/GVO40/GVO40-N11-51.csv",
        "TestData/0112/GVO40/GVO40-N11-53.5.csv",
        "TestData/0112/GVO40/GVO40-N11-56.csv",
        "TestData/0112/GVO40/GVO40-N11-58.5.csv",
        "TestData/0112/GVO40/GVO40-N11-61.csv",
        "TestData/0112/GVO40/GVO40-N11-63.5.csv",
        "TestData/0112/GVO40/GVO40-N11-66.csv",
        "TestData/0112/GVO40/GVO40-N11-68.5.csv",
        "TestData/0112/GVO40/GVO40-N11-71.csv",
        "TestData/0112/GVO40/GVO40-N11-73.5.csv",
        "TestData/0112/GVO40/GVO40-N11-76.csv",

        "TestData/0112/GVO50/GVO50-N11-46.csv",
        "TestData/0112/GVO50/GVO50-N11-48.5.csv",
        "TestData/0112/GVO50/GVO50-N11-51.csv",
        "TestData/0112/GVO50/GVO50-N11-53.5.csv",
        "TestData/0112/GVO50/GVO50-N11-56.csv",
        "TestData/0112/GVO50/GVO50-N11-58.5.csv",
        "TestData/0112/GVO50/GVO50-N11-61.csv",
        "TestData/0112/GVO50/GVO50-N11-63.5.csv",
        "TestData/0112/GVO50/GVO50-N11-66.csv",
        "TestData/0112/GVO50/GVO50-N11-68.5.csv",
        "TestData/0112/GVO50/GVO50-N11-71.csv",
        "TestData/0112/GVO50/GVO50-N11-73.5.csv",
        "TestData/0112/GVO50/GVO50-N11-76.csv",

        "TestData/0112/GVO60/GVO60-N11-46.csv",
        "TestData/0112/GVO60/GVO60-N11-48.5.csv",
        "TestData/0112/GVO60/GVO60-N11-51.csv",
        "TestData/0112/GVO60/GVO60-N11-53.5.csv",
        "TestData/0112/GVO60/GVO60-N11-56.csv",
        "TestData/0112/GVO60/GVO60-N11-58.5.csv",
        "TestData/0112/GVO60/GVO60-N11-61.csv",
        "TestData/0112/GVO60/GVO60-N11-63.5.csv",
        "TestData/0112/GVO60/GVO60-N11-66.csv",
        "TestData/0112/GVO60/GVO60-N11-68.5.csv",
        "TestData/0112/GVO60/GVO60-N11-71.csv",
        "TestData/0112/GVO60/GVO60-N11-73.5.csv",
        "TestData/0112/GVO60/GVO60-N11-76.csv",

        "TestData/0112/GVO70/GVO70-N11-46.csv",
        "TestData/0112/GVO70/GVO70-N11-48.5.csv",
        "TestData/0112/GVO70/GVO70-N11-51.csv",
        "TestData/0112/GVO70/GVO70-N11-53.5.csv",
        "TestData/0112/GVO70/GVO70-N11-56.csv",
        "TestData/0112/GVO70/GVO70-N11-58.5.csv",
        "TestData/0112/GVO70/GVO70-N11-61.csv",
        "TestData/0112/GVO70/GVO70-N11-63.5.csv",
        "TestData/0112/GVO70/GVO70-N11-66.csv",
        "TestData/0112/GVO70/GVO70-N11-68.5.csv",
        "TestData/0112/GVO70/GVO70-N11-71.csv",
        "TestData/0112/GVO70/GVO70-N11-73.5.csv",
        "TestData/0112/GVO70/GVO70-N11-76.csv",

        "TestData/0112/GVO80/GVO80-N11-46.csv",
        "TestData/0112/GVO80/GVO80-N11-48.5.csv",
        "TestData/0112/GVO80/GVO80-N11-51.csv",
        "TestData/0112/GVO80/GVO80-N11-53.5.csv",
        "TestData/0112/GVO80/GVO80-N11-56.csv",
        "TestData/0112/GVO80/GVO80-N11-58.5.csv",
        "TestData/0112/GVO80/GVO80-N11-61.csv",
        "TestData/0112/GVO80/GVO80-N11-63.5.csv",
        "TestData/0112/GVO80/GVO80-N11-66.csv",
        "TestData/0112/GVO80/GVO80-N11-68.5.csv",
        "TestData/0112/GVO80/GVO80-N11-71.csv",
        "TestData/0112/GVO80/GVO80-N11-73.5.csv",
        "TestData/0112/GVO80/GVO80-N11-76.csv",

        "TestData/0112/GVO90/GVO90-N11-46.csv",
        "TestData/0112/GVO90/GVO90-N11-48.5.csv",
        "TestData/0112/GVO90/GVO90-N11-51.csv",
        "TestData/0112/GVO90/GVO90-N11-53.5.csv",
        "TestData/0112/GVO90/GVO90-N11-56.csv",
        "TestData/0112/GVO90/GVO90-N11-58.5.csv",
        "TestData/0112/GVO90/GVO90-N11-61.csv",
        "TestData/0112/GVO90/GVO90-N11-63.5.csv",
        "TestData/0112/GVO90/GVO90-N11-66.csv",
        "TestData/0112/GVO90/GVO90-N11-68.5.csv",
        "TestData/0112/GVO90/GVO90-N11-71.csv",
        "TestData/0112/GVO90/GVO90-N11-73.5.csv",
        "TestData/0112/GVO90/GVO90-N11-76.csv",

        "TestData/0112/GVO100/GVO100-N11-46.csv",
        "TestData/0112/GVO100/GVO100-N11-48.5.csv",
        "TestData/0112/GVO100/GVO100-N11-51.csv",
        "TestData/0112/GVO100/GVO100-N11-53.5.csv",
        "TestData/0112/GVO100/GVO100-N11-56.csv",
        "TestData/0112/GVO100/GVO100-N11-58.5.csv",
        "TestData/0112/GVO100/GVO100-N11-61.csv",
        "TestData/0112/GVO100/GVO100-N11-63.5.csv",
        "TestData/0112/GVO100/GVO100-N11-66.csv",
        "TestData/0112/GVO100/GVO100-N11-68.5.csv",
        "TestData/0112/GVO100/GVO100-N11-71.csv",
        "TestData/0112/GVO100/GVO100-N11-73.5.csv",
        "TestData/0112/GVO100/GVO100-N11-76.csv",

        "TestData/0112/GVO110/GVO110-N11-46.csv",
        "TestData/0112/GVO110/GVO110-N11-48.5.csv",
        "TestData/0112/GVO110/GVO110-N11-51.csv",
        "TestData/0112/GVO110/GVO110-N11-53.5.csv",
        "TestData/0112/GVO110/GVO110-N11-56.csv",
        "TestData/0112/GVO110/GVO110-N11-58.5.csv",
        "TestData/0112/GVO110/GVO110-N11-61.csv",
        "TestData/0112/GVO110/GVO110-N11-63.5.csv",
        "TestData/0112/GVO110/GVO110-N11-66.csv",
        "TestData/0112/GVO110/GVO110-N11-68.5.csv",
        "TestData/0112/GVO110/GVO110-N11-71.csv",
        "TestData/0112/GVO110/GVO110-N11-73.5.csv",
        "TestData/0112/GVO110/GVO110-N11-76.csv",

        "TestData/0112/GVO120/GVO120-N11-46.csv",
        "TestData/0112/GVO120/GVO120-N11-48.5.csv",
        "TestData/0112/GVO120/GVO120-N11-51.csv",
        "TestData/0112/GVO120/GVO120-N11-53.5.csv",
        "TestData/0112/GVO120/GVO120-N11-56.csv",
        "TestData/0112/GVO120/GVO120-N11-58.5.csv",
        "TestData/0112/GVO120/GVO120-N11-61.csv",
        "TestData/0112/GVO120/GVO120-N11-63.5.csv",
        "TestData/0112/GVO120/GVO120-N11-66.csv",
        "TestData/0112/GVO120/GVO120-N11-68.5.csv",
        "TestData/0112/GVO120/GVO120-N11-71.csv",
        "TestData/0112/GVO120/GVO120-N11-73.5.csv",
        "TestData/0112/GVO120/GVO120-N11-76.csv"
    ]

    rpm =[
        1038.116809,
        1092.337521,
        1149.434612,
        1204.536565,
        1262.285778,
        1319.15182,
        1373.619872,
        1431.940595,
        1489.24671,
        1545.876375,
        1600.057927,
        1664.161312,
        1716.820917,
        1033.116871,
        1084.989012,
        1143.295687,
        1200.379427,
        1258.812982,
        1313.951194,
        1372.220575,
        1426.467594,
        1484.164046,
        1541.125068,
        1599.206528,
        1658.256129,
        1715.652731,
        1018.767957,
        1076.764328,
        1133.388419,
        1191.187564,
        1246.857053,
        1303.351305,
        1358.944418,
        1416.260033,
        1473.932867,
        1528.556422,
        1586.102371,
        1647.505613,
        1702.630917,
        1009.791736,
        1063.198528,
        1116.853822,
        1175.991295,
        1231.961527,
        1291.027373,
        1346.750064,
        1400.594101,
        1458.132994,
        1514.681602,
        1576.455672,
        1631.864906,
        1687.265859,
        996.701567,
        1051.937658,
        1108.773669,
        1162.075976,
        1220.861233,
        1276.568651,
        1333.503749,
        1389.35531,
        1446.612393,
        1501.016777,
        1557.831342,
        1617.895294,
        1679.84072,
        983.964897,
        1037.387992,
        1091.553253,
        1146.792744,
        1203.760191,
        1261.737689,
        1315.554003,
        1373.398519,
        1427.857713,
        1486.790961,
        1545.200763,
        1602.170255,
        1664.749386,
        973.019724,
        1022.540792,
        1073.939227,
        1131.823796,
        1185.779703,
        1245.229149,
        1298.119872,
        1354.461314,
        1410.943691,
        1471.255699,
        1524.885569,
        1583.551765,
        1648.853696,
        955.720046,
        1012.173808,
        1065.361365,
        1119.532787,
        1173.22665,
        1229.235262,
        1285.178729,
        1338.069452,
        1396.50756,
        1449.538512,
        1510.168655,
        1567.629054,
        1628.983584,
        949.332617,
        998.998282,
        1052.501343,
        1106.602361,
        1162.516502,
        1217.285557,
        1271.602662,
        1323.791651,
        1383.821144,
        1435.966938,
        1497.427816,
        1552.361805,
        1617.405351,
        935.966286,
        984.973922,
        1041.91737,
        1090.237645,
        1147.153366,
        1198.306121,
        1253.08968,
        1306.866204,
        1362.590562,
        1421.285643,
        1476.940108,
        1533.274999,
        1598.45621,
    ]


    sensor_index = 1
    blade_num = 17

    OA_vaules = []
    fn_values = []
    fn2_values = []
    fz_values = []
    mean_values = []

    for index in range(len(filenames)):

        filename = filenames[index]

        fn = rpm[index]/60
        fn2 = fn * 2
        fz = fn * blade_num

        print("loading ", filename)
        with open(filename, 'r') as file:
            data = file.readlines()[1:]
        loaded_data = [list(map(float, line.strip().split(','))) for line in data]

        time = [item[0] for item in loaded_data]
        data = [item[sensor_index] for item in loaded_data]

        sampling_time = time[-1] - time[0]

        mean_values.append(np.mean(np.array(data)))

        #OA_vaules.append(get_OA_value(np.array(data)))
        #fn_values.append(get_f_band_value(np.array(data),sampling_time,fn))
        #fn2_values.append(get_f_band_value(np.array(data),sampling_time,fn2))
        #fz_values.append(get_f_band_value(np.array(data),sampling_time,fz))


    print("mean_values:")
    [print(float(item)) for item in mean_values]
    print("")

    #print("OA_values:")
    #[print(float(item)) for item in OA_vaules]
    #print("")
    #print("fn_values:")
    #[print(float(item)) for item in fn_values]
    #print("")
    #print("fn2_values:")
    #[print(float(item)) for item in fn2_values]
    #print("")
    #print("fz_values:")
    #[print(float(item)) for item in fz_values]