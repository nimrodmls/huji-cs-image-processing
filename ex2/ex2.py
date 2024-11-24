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

def visualize_spectrogram(in_signal: np.array, sample_rate: int, dest_filename: str):
    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    # sft_alg.spectrogram calculates abs**2 of the given STFT, then taking the log for better visualization
    plt.imshow(np.log(SFT.spectrogram(in_signal) + 1), aspect='auto', origin='lower', cmap='magma', extent=SFT.extent(in_signal.shape[0]))
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(dest_filename)
    plt.close()

def add_constant_frequency(in_audio, sample_rate, frequency, amplitude):
    """
    """    
    in_data_float = in_audio.astype(np.float64)
    
    # Adding the frequency
    t = np.arange(len(in_data_float)) / sample_rate
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    out_data = in_data_float + sine_wave
    
    # Dealing with overflow
    out_data = np.clip(out_data, -32768, 32767).astype(np.int16)
    
    # Create spectrogram
    visualize_spectrogram(out_data, sample_rate, "original_task1_spec.png")

    return out_data

def detect_constant_frequency(input_file, frequency, threshold):
    """
    """
    # Read the input audio file
    sample_rate, in_data = wavfile.read(input_file)
    
    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    stft_ret = SFT.stft(in_data)
    print(np.abs(stft_ret[np.abs(SFT.f - frequency).argmin()]).mean())

def main():
    sample_rate, in_data = wavfile.read('inputs\\task1.wav')
    #in_data_float = in_data.astype(np.float64) / 32768.0
    
    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    # t = np.linspace(0, len(in_data) / sample_rate, len(in_data_float), endpoint=False)
    # in_data_float += 0.01 * np.sin(2 * np.pi * np.min(SFT.f - 18000) * t)

    stft_ret = SFT.stft(in_data) # Frequency x Time

    visualize_spectrogram(SFT, in_data, "original_task1_spec.png")
    watermark = np.random.uniform(-np.pi, np.pi, size=stft_ret.shape)
    freq_threshold = 18000
    high_freq_indices = SFT.f > freq_threshold
    np.angle(stft_ret[high_freq_indices, :])
    stft_ret[high_freq_indices] = 10 * np.exp(1j * stft_ret[high_freq_indices]) # np.ones(stft_ret.shape[1]) * (0.1 * np.exp(1j * ))

    new_data = SFT.istft(stft_ret, k1=in_data.shape[0])
    new_data_ret = SFT.stft(new_data)
    visualize_spectrogram(SFT, new_data.astype(np.int16), 'task1_watermarked.png')

    wavfile.write('inputs\\task1_out.wav', sample_rate, new_data.astype(np.int16))
    # test watermark
    sample_rate, out_data = wavfile.read('inputs\\task1_out.wav')
    visualize_spectrogram(SFT, out_data, 'task1_out.png')
    stft_ret_out = SFT.stft(out_data) # Frequency x Time
    plt.plot(SFT.spectrogram(out_data)[high_freq_indices].mean(axis=1))
    plt.show()
    pass

def task1():
    """
    """
    amplitude = 1000  # Amplitude for the added frequency

    sample_rate, in_audio = wavfile.read('inputs\\task1.wav')

    # Creating a good watermark
    good_frequency = 18000  # The added frequency in Hz - Choosing a high frequency to avoid human hearing
    out_audio = add_constant_frequency(in_audio, sample_rate, good_frequency, amplitude)
    wavfile.write('task1_good_watermark.wav', sample_rate, out_audio)

    # Creating a bad watermark
    bad_frequency = 1000  # The added frequency in Hz - Choosing a low frequency to be heard
    out_audio = add_constant_frequency(in_audio, sample_rate, bad_frequency, amplitude)
    wavfile.write('task1_bad_watermark.wav', sample_rate, out_audio)

def task2():
    """
    """
    watermarked_files = ['inputs\\0_watermarked.wav', 'inputs\\1_watermarked.wav', 'inputs\\2_watermarked.wav',
                         'inputs\\3_watermarked.wav', 'inputs\\4_watermarked.wav', 'inputs\\5_watermarked.wav',
                         'inputs\\6_watermarked.wav', 'inputs\\7_watermarked.wav', 'inputs\\8_watermarked.wav',]
    for file in watermarked_files:
        sample_rate, in_audio = wavfile.read(file)
        visualize_spectrogram(in_audio, sample_rate, file.replace('.wav', '.png'))

if __name__ == "__main__":
    #main()
    
    task1()
    task2()