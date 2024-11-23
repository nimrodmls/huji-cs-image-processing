import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows, stft


### STFT Hyper Parameters ###

# STFT Window
window_size = 512
gaussian_std = 10
window = windows.hann(window_size)#windows.gaussian(window_size, gaussian_std)

# STFT Parameters
stft_hop = 32#window_size // 2

def visualize_spectrogram(sft_alg: ShortTimeFFT, data: np.array, name):
    # sft_alg.spectrogram calculates abs**2 of the given STFT, then taking the log for better visualization
    plt.imshow(np.log(sft_alg.spectrogram(data) + 1), aspect='auto', origin='lower', cmap='magma', extent=sft_alg.extent(data.shape[0]))
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(name)
    plt.close()

def main():
    sample_rate, in_data = wavfile.read('inputs\\task1.wav')
    
    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    stft_ret = SFT.stft(in_data) # Frequency x Time

    #visualize_spectrogram(SFT, data)
    #watermark = np.random.uniform(-np.pi, np.pi, size=stft_ret.shape)
    freq_threshold = 18000
    high_freq_indices = SFT.f > freq_threshold
    stft_ret[high_freq_indices, :] = np.ones(stft_ret.shape[1]) * (500 + 500j)

    new_data = SFT.istft(stft_ret, k1=in_data.shape[0])
    # visualize_spectrogram(SFT, new_data, 'task1_watermarked.png')

    wavfile.write('inputs\\task1_out.wav', sample_rate, new_data.astype(np.int16))
    # test watermark
    sample_rate, out_data = wavfile.read('inputs\\task1_out.wav')
    #visualize_spectrogram(SFT, out_data, 'task1_out.png')
    stft_ret_out = SFT.stft(out_data) # Frequency x Time
    
    pass




if __name__ == "__main__":
    main()