import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows, stft, find_peaks
from scipy.fft import fft, fftfreq


### STFT Hyper Parameters ###

# STFT Window
window_size = 1024
window = windows.hann(window_size)

# STFT Parameters
stft_hop = 32

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
    
    ### Hyperparameters

    # {Distance Between Peaks: Applicable Watermark Category}
    peaks_distancing = {220: 1, 155: 2, 120: 3}
    # The frequency threshold from which the watermark starts (mostly without noise, in the given examples)
    freq_threshold = 19500

    # Categorizing each of the watermarked files
    for file in watermarked_files:

        # Reading the watermarked file and taking its spectrogram (squared absolute values of stft)
        sample_rate, in_audio = wavfile.read(file)
        SFT = ShortTimeFFT(window, fs=sample_rate, hop=512, mfft=window_size*2, scale_to='magnitude')
        stft_ret = SFT.spectrogram(in_audio)
        high_freq_indices = SFT.f > freq_threshold

        # Taking the mean on the amplitudes of the first 27 high frequencies,
        # those were identified as (part of) the watermark
        a = np.abs(stft_ret[high_freq_indices][:27]).T.mean(axis=1).T

        t_secs = in_audio.shape[0] / sample_rate # Time in seconds of the audio
        t_slices_per_sec = a.shape[0] / t_secs # Spectrogram time slices per second

        # Classification of the watermark
        for dist, cat in peaks_distancing.items():
            peaks, _ = find_peaks(a, distance=dist)
            # The frequency in time slices between peaks
            t_peak_slice_freq = np.diff(peaks)
            # The difference between the frequency in time slices between peaks
            peaks_diff = np.abs(np.diff(t_peak_slice_freq)) 
            identified_freq = 1 / (t_peak_slice_freq.mean() / t_slices_per_sec)
            if (peaks_diff < 40).sum() == peaks_diff.shape[0]:
               print(f'{file} is category {cat} - Frequency: {identified_freq}')
               break
        
        # Visualizations - TODO: delete later
        # plt.plot(a)
        # plt.title(file)
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude')
        # plt.savefig(file.replace('.wav', '_amps.png'))
        # plt.close()
        #visualize_spectrogram(in_audio, sample_rate, file.replace('.wav', '.png'))

def task3():
    """
    """
    pass

if __name__ == "__main__":
    #main()
    
    # task1()
    task2()
    task3()