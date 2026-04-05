#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AllPass Analysis
"""

import sys
import os

# Add parent folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverb_classes import AllPass
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr


# ---------------------------------------------------------
# 1) Impulse Response Test
# ---------------------------------------------------------

Fs = 48000
N = 24000

x = np.zeros(N, dtype=np.float32)
x[0] = 1.0

x_imp = x

# AllPass(max_delay_sec, delay_ms, gAll)
ap = AllPass(1, 50, 0.7)
y_imp = np.zeros(N)

for n in range(N):
    y_imp[n] = ap.process(x_imp[n])

t_ms = (np.arange(N) / Fs) * 1000


# --- Plot full impulse response (0–100 ms) ---
plt.figure(figsize=(12,4))
plt.plot(t_ms, y_imp)
plt.title("AllPass Impulse Response (delay = 50 ms, g = 0.7)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 450)
plt.ylim(-1.2, 1.2)
plt.axvline(50, color='red', linestyle='--')
plt.show()


# --- Zoom 1: show the -g spike at 0 ms ---
plt.figure(figsize=(12,4))
plt.plot(t_ms, y_imp)
plt.title("Zoom: First 0.1 ms (showing -g spike)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 0.1)
plt.ylim(-1.2, 1.2)
plt.show()



# ---------------------------------------------------------
# 2) Spectrogram for AllPass with g = 0.3
# ---------------------------------------------------------

# Load audio
x, sr = lr.load("IR48k.wav")
x = x.astype(np.float32)
x = x / np.max(np.abs(x))

ap1 = AllPass(1, 50, 0.3)

y1 = np.zeros_like(x)
for n in range(len(x)):
    y1[n] = ap1.process(x[n])

# STFT
X1 = lr.stft(y1, n_fft=4096, hop_length=64)
X1_db = 20 * np.log10(np.abs(X1) + 1e-12)

# Time axis: frame_time = frame_index * (hop_length / sr)
time_axis = np.arange(X1_db.shape[1]) * 64 / sr

# Frequency axis
freq_axis = np.linspace(0, sr/2, X1_db.shape[0])

plt.figure(figsize=(10, 6))
plt.imshow(
    X1_db,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title("AllPass Spectrogram (g = 0.3)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 0.2)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 3) Spectrogram for AllPass with g = 0.7
# ---------------------------------------------------------

ap2 = AllPass(1, 50, 0.7)

y2 = np.zeros_like(x)
for n in range(len(x)):
    y2[n] = ap2.process(x[n])

# STFT
X2 = lr.stft(y2, n_fft=4096, hop_length=64)
X2_db = 20 * np.log10(np.abs(X2) + 1e-12)

# Time axis
time_axis = np.arange(X2_db.shape[1]) * 64 / sr

# Frequency axis
freq_axis = np.linspace(0, sr/2, X2_db.shape[0])

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
plt.title("AllPass Spectrogram (g = 0.7)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 0.2)
plt.tight_layout()
plt.show()
