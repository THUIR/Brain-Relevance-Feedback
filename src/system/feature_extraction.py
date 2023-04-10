import numpy as np
from scipy.integrate import simps
import math
from  sklearn import preprocessing
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper

FREQ_BANDS = {
    "delta": [0.5, 4],   # 1-3 
    "theta": [4, 8],     # 4-7
    "alpha": [8, 13],    # 8-12
    "beta": [13, 30],    # 13-30
    "gamma": [25,50]
}

def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = min((2 / low) * sf, len(data))

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def bandspower(data, sf, bands, method='welch', window_sec=None, relative=False):
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = min((2 / 0.5) * sf, len(data))

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    re = []
    for band in bands:
        band = np.asarray(band)
        low, high = band

        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        re.append(math.log(bp))
    return re

def get_bp(raw, data_sample):
    fs = []

    for i in range(data_sample):
        tmp_data = raw[:, i * 1000 : i * 1000 + 1000]
        tmp_fs = []
        for channel_id in range(tmp_data.shape[0]):
            # tmp_feature = bandspower(tmp_data[channel_id], 1000, list(FREQ_BANDS.values()))
            tmp_feature = []
            for band in FREQ_BANDS.values():
                tmp_feature.append(math.log(bandpower(tmp_data[channel_id], 1000, band)))     
            tmp_fs.append(tmp_feature)
        fs.append(tmp_fs)
    return fs

selected_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

def re_reference(x,):
    reference = np.mean(x[[32,42],:], axis = 0)
    return x - reference

def select_channles(x):
    return x[selected_index,:]

# 提取特征，数据分段
def preprocessed(x, normalized = True, data_sample = None):
    x = re_reference(x)
    x = select_channles(x)
    x = np.array(x)
    if data_sample == None: 
        data_sample = int(x.shape[1] / 1000)
    if 0 == data_sample:
        return None
    if normalized:
        my_std = preprocessing.StandardScaler()
        try:
            x = get_bp(x, data_sample)
        except Exception as e:
            with open('exception.txt', 'a') as f:
                f.write(str(e))
            return None
        for i in range(len(x)):
            x[i] = my_std.fit_transform(x[i]).tolist()
        return x
    else:
        return get_bp(x, data_sample).tolist()

