import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows, find_peaks
from scipy.fft import fft, fftfreq

task2_watermarked_files = ['inputs\\0_watermarked.wav', 'inputs\\1_watermarked.wav', 'inputs\\2_watermarked.wav',
                         'inputs\\3_watermarked.wav', 'inputs\\4_watermarked.wav', 'inputs\\5_watermarked.wav',
                         'inputs\\6_watermarked.wav', 'inputs\\7_watermarked.wav', 'inputs\\8_watermarked.wav',]

### STFT Hyper Parameters ###

# STFT Window
window_size = 1024
window = windows.hann(window_size)

# STFT Parameters
stft_hop = 32

def visualize_spectrogram(in_signal: np.array, sample_rate: int, dest_filename: str, title: str=''):
    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    plt.imshow(np.log(SFT.spectrogram(in_signal) + 1), aspect='auto', origin='lower', cmap='magma', extent=SFT.extent(in_signal.shape[0]))
    plt.title(title)
    plt.colorbar(label='Magnitude [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.savefig(dest_filename)
    plt.close()

def visualize_task2_spectrograms():
    """
    """
    audio_files = [(wavfile.read(file), file) for file in task2_watermarked_files]    
    sample_rate = audio_files[0][0][0] # Assuming the sampling rate is the same for each file

    SFT = ShortTimeFFT(window, stft_hop, sample_rate, mfft=window_size*2, scale_to='magnitude')
    fig, axs = plt.subplots(3, 3, figsize=(20, 20), sharex=True)
    for i in range(3):
        for j, (in_signal, title) in enumerate(audio_files[i*3:i*3+3]):
            axs[i][j%3].imshow(np.log(SFT.spectrogram(in_signal[1]) + 1), aspect='auto', origin='lower', cmap='viridis', extent=SFT.extent(in_signal[1].shape[0]))
            axs[i][j%3].set_title(title)
            axs[i][j%3].set_ylabel('Frequency [Hz]')
            axs[i][j%3].set_xlabel('Time [s]')
    plt.savefig("task2_spectrograms.png")
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
    visualize_spectrogram(in_audio, sample_rate, "original_task1_spec.png", "Original Audio")

    # Creating a good watermark
    good_frequency = 18000  # The added frequency in Hz - Choosing a high frequency to avoid human hearing
    out_audio = add_constant_frequency(in_audio, sample_rate, good_frequency, amplitude)
    wavfile.write('task1_good_watermark.wav', sample_rate, out_audio)
    visualize_spectrogram(out_audio, sample_rate, "good_task1_spec.png", "Good Watermark")

    # Creating a bad watermark
    bad_frequency = 1000  # The added frequency in Hz - Choosing a low frequency to be heard
    out_audio = add_constant_frequency(in_audio, sample_rate, bad_frequency, amplitude)
    wavfile.write('task1_bad_watermark.wav', sample_rate, out_audio)
    visualize_spectrogram(out_audio, sample_rate, "bad_task1_spec.png", "Bad Watermark")

def task2():
    """
    """
    ### Hyperparameters
    # {Distance Between Peaks: Applicable Watermark Category}
    peaks_distancing = {220: 1, 155: 2, 120: 3}
    # The frequency threshold from which the watermark starts (mostly without noise, in the given examples)
    freq_threshold = 19500

    visualize_task2_spectrograms()

    # Categorizing each of the watermarked files
    for file in task2_watermarked_files:

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

def task3():
    """
    """
    sample_rate1, in_audio1 = wavfile.read('inputs\\task3_watermarked_method1.wav')
    visualize_spectrogram(in_audio1, sample_rate1, "task3_watermarked_method1_spec.png", "Method 1 Watermark")
    sample_rate2, in_audio2 = wavfile.read('inputs\\task3_watermarked_method2.wav')
    visualize_spectrogram(in_audio2, sample_rate2, "task3_watermarked_method2_spec.png", "Method 2 Watermark")

    # File 1 has a lower sample rate, so we match the sample rate of file 2 (it's a factor of 4)
    # for simpler spectrogram analysis
    SFT = ShortTimeFFT(window, fs=sample_rate2, hop=stft_hop, mfft=window_size*2, scale_to='magnitude')

    # Extracting some of the watermark frequencies
    freq_threshold = (17500, 20000)
    high_freq_indices = np.logical_and(SFT.f > freq_threshold[0], SFT.f < freq_threshold[1])

    audio1_stft = SFT.spectrogram(in_audio1)
    audio2_stft = SFT.spectrogram(in_audio2)
    
    # Extracting the amplitudes of the watermark frequencies
    audio1_stft = np.abs(audio1_stft[high_freq_indices]).mean(axis=0)
    audio2_stft = np.abs(audio2_stft[high_freq_indices]).mean(axis=0)

    # The audio with a more "faint" watermark is the audio sped up / slowed down in the frequency domain
    # The audio with a more "clear" watermark is the audio sped up / slowed down in the time domain
    if audio1_stft.max() > audio2_stft.max():
        print(f'File 1 - Time Domain, File 2 - Frequency Domain, factor {sample_rate1 / sample_rate2}')
    else:
        print(f'File 2 - Time Domain, File 1 - Frequency Domain, factor {sample_rate2 / sample_rate1}')

if __name__ == "__main__":    
    task1()
    task2()
    task3()