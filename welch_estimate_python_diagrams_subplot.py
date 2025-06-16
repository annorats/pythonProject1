import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Generate signal
fs = 500
T = 2
N = T * fs
t = np.linspace(0, T, N, endpoint=False)
np.random.seed(0)

signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
signal += np.random.normal(0, 1, t.shape)

# FFT of full signal
fft_full = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/fs)

# Define windows
window_size = 256
start_1 = 0
start_2 = window_size // 2
section_1 = signal[start_1:start_1 + window_size]
section_2 = signal[start_2:start_2 + window_size]

# FFTs of windows
fft_1 = np.fft.fft(section_1)
freqs_1 = np.fft.fftfreq(window_size, 1/fs)

fft_2 = np.fft.fft(section_2)
freqs_2 = np.fft.fftfreq(window_size, 1/fs)

# Welch estimate
f_welch, Pxx = welch(signal, fs=fs, window='hann', nperseg=window_size, noverlap=window_size//2)

# Create subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()

# 1. Time-domain signal
axes[0].plot(t, signal)
axes[0].set_title("1. Random Signal in Time Domain")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# 2. FFT of full signal (normalized)
axes[1].semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N)
axes[1].set_title("2. FFT Over Entire Signal (Normalized)")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Power")
axes[1].grid()

# 4. First window highlighted
axes[2].plot(t, signal, label='Signal')
axes[2].axvspan(t[start_1], t[start_1 + window_size], color='red', alpha=0.3, label='First Window')
axes[2].set_title("4. Signal with First Window Highlighted")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Amplitude")
axes[2].legend()
axes[2].grid()

# 5. FFT of first window
axes[3].plot(freqs_1[:window_size//2], np.abs(fft_1[:window_size//2])**2)
axes[3].set_title("5. FFT of First Window")
axes[3].set_xlabel("Frequency [Hz]")
axes[3].set_ylabel("Power")
axes[3].grid()

# 6. First + second (overlapping) windows
axes[4].plot(t, signal, label='Signal')
axes[4].axvspan(t[start_1], t[start_1 + window_size], color='red', alpha=0.3, label='First Window')
axes[4].axvspan(t[start_2], t[start_2 + window_size], color='blue', alpha=0.3, label='Second Window')
axes[4].set_title("6. Two Overlapping Windows")
axes[4].set_xlabel("Time [s]")
axes[4].set_ylabel("Amplitude")
axes[4].legend()
axes[4].grid()

# 7. FFT of second window
axes[5].plot(freqs_2[:window_size//2], np.abs(fft_2[:window_size//2])**2)
axes[5].set_title("7. FFT of Second Overlapping Window")
axes[5].set_xlabel("Frequency [Hz]")
axes[5].set_ylabel("Power")
axes[5].grid()

# 8. Welch estimate
axes[6].semilogy(f_welch, Pxx)
axes[6].set_title("8. Welch Estimate")
axes[6].set_xlabel("Frequency [Hz]")
axes[6].set_ylabel("Power")
axes[6].grid()

# 9. Compare Welch to full FFT
axes[7].semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N, label='FFT (Whole Signal)')
axes[7].semilogy(f_welch, Pxx, label='Welch Estimate', linestyle='--')
axes[7].set_title("9. Welch vs Full FFT")
axes[7].set_xlabel("Frequency [Hz]")
axes[7].set_ylabel("Power")
axes[7].legend()
axes[7].grid()

# Hide any unused subplots (8 and 9 used, 10 is unused)
for ax in axes[8:]:
    ax.axis("off")

plt.tight_layout()
plt.show()
