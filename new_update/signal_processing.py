import numpy as np
from scipy import signal


class Signal_processing():
    def __init__(self):
        self.a = 1

    def extract_color(self, ROIs):
        """
        extract average value of green color from ROIs
        """
        # r = np.mean(ROI[:,:,0])
        g = []
        for ROI in ROIs:
            g.append(np.mean(ROI[:, :, 1]))
        # b = np.mean(ROI[:,:,2])
        # return r, g, b
        output_val = np.mean(g)
        return output_val

    def normalization(self, data_buffer):
        """
        normalize the input data buffer
        """
        # normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
        return data_buffer / np.linalg.norm(data_buffer)

    def signal_detrending(self, data_buffer):
        """
        remove overall trending
        """
        return signal.detrend(data_buffer)

    def interpolation(self, data_buffer, times):
        """
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        """
        L = len(data_buffer)
        even_times = np.linspace(times[0], times[-1], L)

        return np.hamming(L) * np.interp(even_times, times, data_buffer)

    def fft(self, data_buffer, fps):
        L = len(data_buffer)

        freqs = float(fps) / L * np.arange(L / 2 + 1)

        freqs_in_minute = 60. * freqs

        raw_fft = np.fft.rfft(data_buffer * 30)
        fft = np.abs(raw_fft) ** 2

        interest_idx = np.where((freqs_in_minute > 50) & (freqs_in_minute < 180))[0]
        print(freqs_in_minute)
        interest_idx_sub = interest_idx[:-1].copy()  # avoid the indexing error
        # pruned = fft[interest_idx]
        # pfreq = freqs_in_minute[interest_idx]

        # freqs_of_interest = pfreq 
        # fft_of_interest = pruned

        return fft[interest_idx_sub], freqs_in_minute[interest_idx_sub]

    def butter_bandpass_filter(self, data_buffer, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_data = signal.lfilter(b, a, data_buffer)

        return filtered_data
