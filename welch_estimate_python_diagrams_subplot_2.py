import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Custom color palette
colors = [
    '#1f77b4',  # blue (main time series)
    '#ff7f0e',  # orange
    '#2ca02c',  # green ← use for first window
    '#d62728',  # red   ← use for second window
    '#9467bd',  # purple
    '#8aff48',  # lime
    '#ff369e'  # pink
]

# Generate sampling signal
#Sampling signal fs
fs = 500
#Time T
T = 2
#Number of total samples N
N = T * fs
#time vector, length N, evenly spaced from 0 to one before T, sampling times
t = np.linspace(0, T, N, endpoint=False)
np.random.seed(0)

#creates a signal of two sine waves, one at 50 Hz amp 1, the other 120 Hz, amp 0.5
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
#adds Gaussian (normal) noise with mean 0 and std 1 to signal making it noisy
signal += np.random.normal(0, 1, t.shape)

#FFT of full signal, converting it from time to frequency domain
fft_full = np.fft.fft(signal)
#finds the corrosponding frequency bins for the fft, from 0 to the Nyquist frequency, and negative frequencies
#because the fft output is symmetric.
freqs = np.fft.fftfreq(N, 1/fs)

# Define window size
window_size = 256
start_1 = 0
start_2 = window_size // 2
section_1 = signal[start_1:start_1 + window_size]
section_2 = signal[start_2:start_2 + window_size]

# FFTs of sections
fft_1 = np.fft.fft(section_1)
freqs_1 = np.fft.fftfreq(window_size, 1/fs)
fft_2 = np.fft.fft(section_2)
freqs_2 = np.fft.fftfreq(window_size, 1/fs)

# Welch estimate, returns welch frequency f_welch, and Power spectral density values PSD
f_welch, PSD = welch(signal, fs=fs, window='hann', nperseg=window_size, noverlap=window_size//2)

# Create subplots
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.flatten()

# 1. Time domain
axes[0].plot(t, signal, color=colors[0])
axes[0].set_title("1. Time-Domain Signal")
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[0].set_xlim(0, T)

# 2. Full FFT
axes[1].semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N, color=colors[1])
axes[1].set_title("2. FFT of Entire Signal (Normalised, Log Scale)")
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Power")
axes[1].grid(True)
axes[1].set_xlim(0, fs/2)

# 3. First window marked
axes[2].plot(t, signal, label='Signal', color=colors[0])
axes[2].axvspan(t[start_1], t[start_1 + window_size], color=colors[5], alpha=0.3, label='First Window')
axes[2].set_title("3. Signal with First Window Highlighted")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Amplitude")
axes[2].legend()
axes[2].grid(True)
axes[2].set_xlim(0, T)

# 4. FFT of first window
axes[3].plot(freqs_1[:window_size//2], np.abs(fft_1[:window_size//2])**2, color=colors[2])
axes[3].set_title("4. FFT of First Window")
axes[3].set_xlabel("Frequency [Hz]")
axes[3].set_ylabel("Power")
axes[3].grid(True)
axes[3].set_xlim(0, fs/2)

# 5. Both windows marked
axes[4].plot(t, signal, label='Signal', color=colors[0])
axes[4].axvspan(t[start_1], t[start_1 + window_size], color=colors[5], alpha=0.3, label='First Window')
axes[4].axvspan(t[start_2], t[start_2 + window_size], color=colors[6], alpha=0.3, label='Second Window')
axes[4].set_title("5. Signal with Two Overlapping Windows")
axes[4].set_xlabel("Time [s]")
axes[4].set_ylabel("Amplitude")
axes[4].legend()
axes[4].grid(True)
axes[4].set_xlim(0, T)

# 6. FFT of second window
axes[5].plot(freqs_2[:window_size//2], np.abs(fft_2[:window_size//2])**2, color=colors[3])
axes[5].set_title("6. FFT of Second Overlapping Window")
axes[5].set_xlabel("Frequency [Hz]")
axes[5].set_ylabel("Power")
axes[5].grid(True)
axes[5].set_xlim(0, fs/2)

# 7. Welch Estimate alone
axes[6].semilogy(f_welch, PSD, color=colors[4])
axes[6].set_title("7. Welch Power Spectral Density Estimate")
axes[6].set_xlabel("Frequency [Hz]")
axes[6].set_ylabel("Power")
axes[6].grid(True)
axes[6].set_xlim(0, fs/2)

# 8. Comparison of Welch vs full FFT
axes[7].semilogy(freqs[:N//2], np.abs(fft_full[:N//2])**2 / N, label='FFT (Whole)', color=colors[1])
axes[7].semilogy(f_welch, PSD, linestyle='--', label='Welch Estimate', color=colors[4])
axes[7].set_title("8. Comparison: Welch vs Full FFT")
axes[7].set_xlabel("Frequency [Hz]")
axes[7].set_ylabel("Power")
axes[7].legend()
axes[7].grid(True)
axes[7].set_xlim(0, fs/2)

# Hide any extra axes
for ax in axes[8:]:
    ax.axis("off")

# Tight layout and global title
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.suptitle("Welch Estimate vs FFT: Step-by-Step Illustration", fontsize=18)

# Save output
fig.savefig("welch_estimate_plot.png", dpi=300, bbox_inches='tight')
fig.savefig("welch_estimate_plot.pdf", bbox_inches='tight')
plt.show()
