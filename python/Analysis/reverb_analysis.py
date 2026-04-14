#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverb Analysis
Armando Iachini
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import sounddevice as sd
import soundfile as sf

# Import your reverb class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reverb_classes import Reverb

Fs = 48000
N = 48000


# ---------------------------------------------------------
# TEST 1 — PURE PRE-DELAY + Comb
# ---------------------------------------------------------

x = np.zeros(N, dtype=np.float32)
x[0] = 1.0   # impulse

reverb = Reverb(
    max_delay_sec=3.0,
    a=0.5,
    g=0.5,
    gAll=0.5,
    numb_Combs=1,
    numb_AllPass=0,
    room_Size=2,
    mix=1.0,          # wet only
    predelay_ms=0  # small room first comb delay = 20 ms → total = 100 ms
)

y = np.zeros_like(x)
for n in range(N):
    y[n] = reverb.process(x[n])

# Time axis in ms
t_ms = (np.arange(N) / Fs) * 1000.0

# Plot with fixed y-axis and markers
plt.figure(figsize=(10, 4))
plt.plot(t_ms, y, marker='o', markersize=3, linestyle='-')
plt.title("Pure Pre-Delay + Comb (Expected spike at 100 ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

# Make spike visible
plt.ylim(-0.1, 1.1)
plt.xlim(0, 200)

plt.show()

# Listen
sd.play(y / (np.max(np.abs(y)) + 1e-12), Fs)
sd.wait()


# ---------------------------------------------------------
# 2) PLOT IMPULSE RESPONSE OF REVERB
# --------------------------------------------------------

# Create impulse
x1 = np.zeros(N, dtype=np.float32)
x1[0] = 1.0

# Create reverb
reverb = Reverb(
    max_delay_sec=3.0,
    a=0.8,
    g=0.5,
    gAll=0.7,
    numb_Combs=8,
    numb_AllPass=4,
    room_Size=2,      # large room
    mix=0.5,
    predelay_ms=20
)

# Process impulse through reverb
y = np.zeros_like(x1)
for n in range(N):
    y[n] = reverb.process(x1[n])

# Time axis in ms
t_ms = (np.arange(N) / Fs) * 1000.0

# Plot IR
plt.figure(figsize=(12, 4))
plt.plot(t_ms, y)
plt.title("Reverb Impulse Response (Room Size = Large, Pre-delay = 20 ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 600)   # zoom into first 600 ms
plt.tight_layout()
plt.show()

# Playback (normalized)
print("Playing impulse response...")
y_norm = y / (np.max(np.abs(y)) + 1e-12)
sd.play(y_norm, Fs)
sd.wait()

#-------------------------------------
# Generate Logarithmic Sweep 
#-------------------------------------

sr = 48000
duration = 3.0
N = int(duration * sr)

"Manual logarithmic sweep (20 Hz → 20 kHz)"
t = np.linspace(0, duration, N)

f1 = 20
f2 = 20000

K = duration / np.log(f2 / f1)
sweep = np.sin(2 * np.pi * f1 * K * (np.exp(t / K) - 1))

sweep = sweep.astype(np.float32)
sweep = sweep / np.max(np.abs(sweep))

plt.figure()
plt.plot(t, sweep)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Logarithmic Sweep (20 Hz → 20 kHz)")
plt.tight_layout()
plt.show()

#Sweep Spect
hop_length = 512
Xstft_sweep = lr.stft(sweep, n_fft=2048, hop_length=hop_length)

Xm_sweep = np.abs(Xstft_sweep)
log_mag_sweep = np.log(Xm_sweep + 1e-8)

freq_axis_sweep = np.linspace(0, sr/2, Xm_sweep.shape[0])
time_axis_sweep = np.arange(Xm_sweep.shape[1]) * hop_length / sr

plt.figure(figsize=(12, 6))
plt.imshow(
    log_mag_sweep,
    aspect='auto',
    origin='lower',
    extent=[time_axis_sweep[0], time_axis_sweep[-1], freq_axis_sweep[0], freq_axis_sweep[-1]]
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram – Original Logarithmic Sweep")
plt.colorbar(label="Log Magnitude")
plt.tight_layout()
plt.show()
#-------------------------------------
#Pass Sweep Through Reverb 
#-------------------------------------

reverb = Reverb(
    max_delay_sec=6.0,
    a=0.99,
    g=0.5,
    gAll=0.7,
    numb_Combs=8,
    numb_AllPass=4,
    room_Size=2,      # large room
    mix=0.7,
    predelay_ms=20
)

y_sweep = np.zeros_like(sweep)
for n in range(N):
    y_sweep[n] = reverb.process(sweep[n])

"Normalise for listening"
y_sweep_norm = y_sweep / (np.max(np.abs(y_sweep)) + 1e-12)


#-------------------------------------
#3) Plot Waveform of Reverb Output "
#-------------------------------------

plt.figure()
plt.plot(t, y_sweep_norm)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Reverb Output – Logarithmic Sweep")
plt.tight_layout()
plt.show()


#-------------------------------------
# STFT Spectrogram 
#-------------------------------------

hop_length = 512
Xstft = lr.stft(y_sweep_norm, n_fft=2048, hop_length=hop_length)

Xm = np.abs(Xstft)
log_mag = np.log(Xm + 1e-8)

freq_axis = np.linspace(0, sr/2, Xm.shape[0])
time_axis = np.arange(Xm.shape[1]) * hop_length / sr

plt.figure(figsize=(12, 6))
plt.imshow(
    log_mag,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram – Reverb Output (Log Sweep)")
plt.colorbar(label="Log Magnitude")
plt.tight_layout()
plt.show()


#-------------------------------------
" 5) Listen to Sweep Through Reverb "
#-------------------------------------

print("Playing sweep through reverb...")
sd.play(y_sweep_norm, sr)
sd.wait()


#-------------------------------------
" 3) Compare Spectrograms: Big Damping vs Small Damping "
#-------------------------------------

# --- Reverb with BIG damping (a = 0.8) ---
reverb_big = Reverb(
    max_delay_sec=3.0,
    a=0.99,          # BIG damping
    g=0.5,
    gAll=0.7,
    numb_Combs=8,
    numb_AllPass=4,
    room_Size=2,
    mix=1,
    predelay_ms=500
)

y_big = np.zeros_like(sweep)
for n in range(N):
    y_big[n] = reverb_big.process(sweep[n])

y_big_norm = y_big / (np.max(np.abs(y_big)) + 1e-12)


# --- Reverb with SMALL damping (a = 0.2) ---
reverb_small = Reverb(
    max_delay_sec=3.0,
    a=0.1,          # SMALL damping
    g=0.5,
    gAll=0.7,
    numb_Combs=8,
    numb_AllPass=4,
    room_Size=2,
    mix=1,
    predelay_ms=500
)

y_small = np.zeros_like(sweep)
for n in range(N):
    y_small[n] = reverb_small.process(sweep[n])

y_small_norm = y_small / (np.max(np.abs(y_small)) + 1e-12)


# --- Compute STFTs ---
hop_length = 512

X_big = lr.stft(y_big_norm, n_fft=2048, hop_length=hop_length)
X_small = lr.stft(y_small_norm, n_fft=2048, hop_length=hop_length)

log_big = np.log(np.abs(X_big) + 1e-8)
log_small = np.log(np.abs(X_small) + 1e-8)

freq_axis = np.linspace(0, sr/2, log_big.shape[0])
time_axis = np.arange(log_big.shape[1]) * hop_length / sr


# --- Plot side-by-side ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.imshow(
    log_big,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
)
plt.title("Spectrogram – BIG Damping (a = 0.99)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Log Magnitude")

plt.subplot(2, 1, 2)
plt.imshow(
    log_small,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
)
plt.title("Spectrogram – SMALL Damping (a = 0.1)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Log Magnitude")

plt.tight_layout()
plt.show()
