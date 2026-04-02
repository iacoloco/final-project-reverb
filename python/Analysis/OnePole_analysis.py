#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:46:49 2026

@author: armandoiachini
"""
import sys
import os

# Add parent folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverb_classes import OnePole
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 2) Spectrogram Comparison for a = 0.1 and a = 0.9 (in dB)
# ---------------------------------------------------------

import librosa as lr

# Load audio
x, sr = lr.load("IR.wav")
x = x.astype(np.float32)
x = x / np.max(np.abs(x))

# Parameters
hop_length = 64
n_fft = 4096

# ---------------------------------------------------------
# Spectrogram for OnePole with a = 0.1
# ---------------------------------------------------------

a1 = 0.1
lp1 = OnePole(a=a1)

# Create buffer 
y1 = np.zeros_like(x)

# Process audio 
for n in range(len(x)):
    y1[n] = lp1.process(x[n])

# STFT 
# n_fft = 4096     # hop_length = 64 gives high time resolution

X1 = lr.stft(y1, n_fft=4096, hop_length=64)

# Convert magnitude to dB
X1_db = 20 * np.log10(np.abs(X1) + 1e-12)

# Build time axis----->frame_time = frame_index * (hop_length / sr)
time_axis = np.arange(X1_db.shape[1]) * 64 / sr

# Build frequency axis (0 Hz → Nyquist)
freq_axis = np.linspace(0, sr/2, X1_db.shape[0])

# Plot spectrogram
plt.figure(figsize=(10, 6))
plt.imshow(
    X1_db,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,   # dynamic range
    vmax=0
)
plt.title("OnePole Spectrogram (a = 0.1)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")

# Zoom into the first 200 ms (where the interesting stuff happens)
plt.xlim(0, 0.2)

plt.tight_layout()
plt.show()



# ---------------------------------------------------------
# Spectrogram for OnePole with a = 0.9
# ---------------------------------------------------------

a2 = 0.9
lp2 = OnePole(a=a2)

# Create output buffer
y2 = np.zeros_like(x)

# Process audio
for n in range(len(x)):
    y2[n] = lp2.process(x[n])

# STFT
X2 = lr.stft(y2, n_fft=4096, hop_length=64)

# Convert to dB
X2_db = 20 * np.log10(np.abs(X2) + 1e-12)

# Time axis (same formula as above)
time_axis = np.arange(X2_db.shape[1]) * 64 / sr   # frame_time = frame_index * (hop_length / sr)

# Frequency axis
freq_axis = np.linspace(0, sr/2, X2_db.shape[0])

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(
    X2_db,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title("OnePole Spectrogram (a = 0.9)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 0.2)
plt.tight_layout()
plt.show()
