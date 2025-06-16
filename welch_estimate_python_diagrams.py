import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 1. Generate a random signal
fs = 500  # Sampling frequency
T = 2     # Duration in seconds
N = T * fs
t = np.linspace(0, T, N, endpoint=False)
np.random.seed(0)

# Signal = 50 Hz sine + 120 Hz sine + noise
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
signal += np.random.normal(0, 1, t.shape)

plt.figure(figsize=(12, 3))
plt.plot(t, signal)
plt.title("1. Random Signal in Time Domain")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.tight_layout()
plt.show()

# 2. FFT over the entire data set
fft_full = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure(figsize=(12, 3))
plt.semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N)
plt.title("2 (Revised). FFT Over Entire Signal (Normalized, Log Scale)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.grid()
plt.tight_layout()
plt.show()

# 3. Loss of precision in time: frequency is OK but time localisation is lost
# This is explained visually and by comparing to windowed sections.

# 4. Highlight a small section at beginning
window_size = 256
start_1 = 0
section_1 = signal[start_1:start_1 + window_size]

plt.figure(figsize=(12, 3))
plt.plot(t, signal, label='Signal')
plt.axvspan(t[start_1], t[start_1 + window_size], color='red', alpha=0.3, label='First Window')
plt.title("4. Original Signal with First Window Highlighted")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 5. FFT of first section
fft_1 = np.fft.fft(section_1)
freqs_1 = np.fft.fftfreq(window_size, 1/fs)

plt.figure(figsize=(12, 3))
plt.plot(freqs_1[:window_size//2], np.abs(fft_1[:window_size//2])**2)
plt.title("5. FFT of First Window")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.grid()
plt.tight_layout()
plt.show()

# 6. Second window, overlapping
start_2 = window_size // 2
section_2 = signal[start_2:start_2 + window_size]

plt.figure(figsize=(12, 3))
plt.plot(t, signal, label='Signal')
plt.axvspan(t[start_1], t[start_1 + window_size], color='red', alpha=0.3, label='First Window')
plt.axvspan(t[start_2], t[start_2 + window_size], color='blue', alpha=0.3, label='Second Window (Overlapping)')
plt.title("6. Original Signal with Two Overlapping Windows")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 7. FFT of second section
fft_2 = np.fft.fft(section_2)
freqs_2 = np.fft.fftfreq(window_size, 1/fs)

plt.figure(figsize=(12, 3))
plt.plot(freqs_2[:window_size//2], np.abs(fft_2[:window_size//2])**2)
plt.title("7. FFT of Second Overlapping Window")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.grid()
plt.tight_layout()
plt.show()

# 8. Welch estimate
f_welch, Pxx = welch(signal, fs=fs, window='hann', nperseg=window_size, noverlap=window_size//2)

# 9. Compare Welch with full FFT
plt.figure(figsize=(12, 4))
plt.semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N, label='FFT (Whole Signal)')
plt.semilogy(f_welch, Pxx, label='Welch Estimate', linestyle='--')
plt.title("9. Welch vs Full FFT")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power (log scale)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
