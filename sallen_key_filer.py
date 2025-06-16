import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data from CSV file
df = pd.read_csv("sallen_key_data.csv")

# Caculate Gain in dB
df["Gain (dB)"] = 20 * np.log10(df["Vout"] / df["Vin"])

# Compute Log10 of Frequency
df["Log10 Frequency"] = np.log10(df["Frequency"])

# Identify region for linear fit (frequencies > 500 kHz)
fit_region = df["Frequency"] > 250
slope, intercept, _, _, _ = linregress(df.loc[fit_region, "Log10 Frequency"], df.loc[fit_region, "Gain (dB)"])

# Generate fitted line
df["Fitted Line"] = slope * df["Log10 Frequency"] + intercept

# Plot data
plt.figure(figsize=(8, 5))
plt.scatter(df["Log10 Frequency"], df["Gain (dB)"], label="Log Frequency vs dB", color='blue', s=10)
plt.plot(df["Log10 Frequency"], df["Fitted Line"], 'r--', label=f'Fitted Linear Line\nGradient = {slope:.2f}')

# Labels and Title
plt.xlabel("Log_10 Frequency (kHz)")
plt.ylabel("dB")
plt.title("Sallen-Key Filter Response with Fitted Linear Line log_10 scale")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.show()
