#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of all DSP classes:
DelayLine, OnePole, AllPass, LPcomb, Reverb
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from reverb_classes import DelayLine, OnePole, AllPass, LPcomb, Reverb

Fs = 48000
N = 20000

# ---------------------------------------------------------
# Helper: generate impulse
# ---------------------------------------------------------
def impulse(N):
    x = np.zeros(N, dtype=np.float32)
    x[0] = 1.0
    return x


# ---------------------------------------------------------
# 1. DelayLine Test   # First Delay at 50 ms 
# ---------------------------------------------------------
x = impulse(N)
delay = DelayLine(max_delay_sec=1, delay_ms=50)
y = np.zeros(N)

for n in range(N):
    y[n] = delay.process(x[n])

t_ms = (np.arange(N) / Fs) * 1000   # convert samples → milliseconds

plt.figure(figsize=(12,4))
plt.plot(t_ms, y)
plt.title("DelayLine Impulse Response (50 ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

# Set ticks every 10 ms
plt.xticks(np.arange(0, t_ms[-1], 10))

# Add a vertical line at the expected delay
plt.axvline(x=50, color='red', linestyle='--', label='Expected 50 ms delay')
plt.legend()

# Limit x-axis to 0–100 ms
plt.xlim(0, 100)

plt.show()




# ---------------------------------------------------------
# 2. OnePole Lowpass Test
# ---------------------------------------------------------

x = impulse(N)
lp = OnePole(a=0.7)
y = np.zeros(N)

for n in range(N):
    y[n] = lp.process(x[n])

t_ms = (np.arange(N) / Fs) * 1000   # convert to ms

plt.figure(figsize=(12,4))
plt.plot(t_ms, y)
plt.title("OnePole Lowpass Impulse Response (a = 0.7)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

# Zoom to first 5 ms (critical!)
plt.xlim(0, 5)

# Tick every 0.5 ms
plt.xticks(np.arange(0, 5.1, 0.5))

plt.show()

# ---------------------------------------------------------
# OnePole Spectrogram Test using IR.wav
# ---------------------------------------------------------



# ---------------------------------------------------------
# Load audio
# ---------------------------------------------------------
x, Fs = sf.read("IR.wav", dtype="float32")

# Ensure mono
if x.ndim > 1:
    x = x[:, 0]

# Normalize
x = x / np.max(np.abs(x))

# ---------------------------------------------------------
# Process through OnePole
# ---------------------------------------------------------
lp1 = OnePole(a=0.1)
y = np.zeros_like(x)

for n in range(len(x)):
    y[n] = lp1.process(x[n])

# ---------------------------------------------------------
# Compute STFT
# ---------------------------------------------------------
S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
S_db = librosa.amplitude_to_db(S, ref=np.max)

# ---------------------------------------------------------
# Plot spectrogram
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
librosa.display.specshow(
    S_db,
    sr=Fs,
    hop_length=256,
    x_axis='time',
    y_axis='log',
    cmap='magma'
)
plt.colorbar(format="%+2.0f dB")
plt.title("OnePole Output Spectrogram (a = 0.7)")
plt.show()


