#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:07:14 2026

@author: armandoiachini
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reverb Analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverb_classes import Reverb  #



Fs = 48000
N = 49000  

# ---------------------------------------------------------
# 1) Impulse response of the reverb
# ---------------------------------------------------------

x= np.zeros(N, dtype=np.float32)
x[0] = 1.0

# Reverb(max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass, room_Size)
# room_Size: 0 = small, 1 = medium, 2 = large
reverb = Reverb(
    max_delay_sec=3.0,
    a=0.3,        # medium damping
    g=0.1,        # medium-long tail
    gAll=0.7,     # strong diffusion
    numb_Combs=6,
    numb_AllPass=3,
    room_Size=1,   # medium room,
    mix = 0.5
)

y = np.zeros_like(x)
for n in range(N):
    y[n] = reverb.process(x[n])

t_ms = (np.arange(N) / Fs) * 1000.0

plt.figure(figsize=(12, 4))
plt.plot(t_ms, y)
plt.title("Reverb Impulse Response (medium room, a=0.3, g=0.7, gAll=0.7)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 600)
plt.tight_layout()
plt.show()

# Zoom early reflections
plt.figure(figsize=(12, 4))
plt.plot(t_ms, y)
plt.title("Reverb Impulse Response - Early Reflections (first 150 ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 150)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2) Spectrogram of IR through the reverb
# ---------------------------------------------------------

x_ir, sr = lr.load("IR48k.wav", sr=Fs)
x_ir = x_ir.astype(np.float32)
x_ir = x_ir / (np.max(np.abs(x_ir)) + 1e-12)

rev = Reverb(
    max_delay_sec=1.0,
    a=0.3,
    g=0.7,
    gAll=0.7,
    numb_Combs=6,
    numb_AllPass=3,
    room_Size=1,
    mix=0.5
)

y_ir = np.zeros_like(x_ir)
for n in range(len(x_ir)):
    y_ir[n] = rev.process(x_ir[n])

X = lr.stft(y_ir, n_fft=4096, hop_length=64)
X_db = 20 * np.log10(np.abs(X) + 1e-12)

time_axis = np.arange(X_db.shape[1]) * 64 / Fs
freq_axis = np.linspace(0, Fs / 2, X_db.shape[0])

plt.figure(figsize=(10, 6))
plt.imshow(
    X_db,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title("Reverb Spectrogram (medium room, a=0.3, g=0.7, gAll=0.7)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, min(3, time_axis[-1]))
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3) Compare damping: light vs strong (a)
# ---------------------------------------------------------

def process_ir_with_reverb(a, g, gAll, room_Size, label):
    rev = Reverb(
        max_delay_sec=1.0,
        a=a,
        g=g,
        gAll=gAll,
        numb_Combs=6,
        numb_AllPass=3,
        room_Size=room_Size,
        mix=0.5
    )
    y = np.zeros_like(x_ir)
    for n in range(len(x_ir)):
        y[n] = rev.process(x_ir[n])
    X = lr.stft(y, n_fft=4096, hop_length=64)
    X_db = 20 * np.log10(np.abs(X) + 1e-12)
    return X_db, label

X_light, label_light = process_ir_with_reverb(a=0.1, g=0.7, gAll=0.7, room_Size=1,
                                              label="light damping (a=0.1)")
X_strong, label_strong = process_ir_with_reverb(a=0.7, g=0.7, gAll=0.7, room_Size=1,
                                                label="strong damping (a=0.7)")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(
    X_light,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_light}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.subplot(1, 2, 2)
plt.imshow(
    X_strong,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_strong}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4) Compare tail length: short vs long (g)
# ---------------------------------------------------------

X_short, label_short = process_ir_with_reverb(a=0.3, g=0.4, gAll=0.7, room_Size=1,
                                              label="short tail (g=0.4)")
X_long, label_long = process_ir_with_reverb(a=0.3, g=0.9, gAll=0.7, room_Size=1,
                                            label="long tail (g=0.9)")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(
    X_short,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_short}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.subplot(1, 2, 2)
plt.imshow(
    X_long,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_long}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5) Compare room sizes: small vs medium vs large
# ---------------------------------------------------------

X_small, label_small = process_ir_with_reverb(a=0.3, g=0.7, gAll=0.7, room_Size=0,
                                              label="small room")
X_medium, label_medium = process_ir_with_reverb(a=0.3, g=0.7, gAll=0.7, room_Size=1,
                                                label="medium room")
X_large, label_large = process_ir_with_reverb(a=0.3, g=0.7, gAll=0.7, room_Size=2,
                                              label="large room")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(
    X_small,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_small}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.subplot(1, 3, 2)
plt.imshow(
    X_medium,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_medium}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, min(3, time_axis[-1]))

plt.subplot(1, 3, 3)
plt.imshow(
    X_large,
    aspect='auto',
    origin='lower',
    extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
    cmap='magma',
    vmin=-80,
    vmax=0
)
plt.title(f"Reverb Spectrogram - {label_large}")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, 6)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 6) LISTEN to the IR processed through the reverb
# ---------------------------------------------------------

import sounddevice as sd


# Load IR48k.wav (already loaded earlier as x_ir)
# But let's ensure it's normalized and float32
audio = x_ir.astype(np.float32)
audio /= np.max(np.abs(audio)) + 1e-12

# Create a reverb instance for listening
rev_listen = Reverb(
    max_delay_sec=1.0,
    a=0.7,        # stronger damping (warmer)
    g=0.4,        # shorter, smoother tail
    gAll=0.5,     # classic diffusion
    numb_Combs=6,
    numb_AllPass=3,
    room_Size=1,
    mix=1.0       # listen to the wet signal only
)


# Process audio sample-by-sample
y_listen = np.zeros_like(audio)
for n in range(len(audio)):
    y_listen[n] = rev_listen.process(audio[n])

# Normalize output to avoid clipping
y_listen /= np.max(np.abs(y_listen)) + 1e-12

# PLAY IT
print("Playing processed IR...")
sd.play(y_listen, Fs)
sd.wait()

# ---------------------------------------------------------
# 7) LISTEN to a real song processed through the reverb
# ---------------------------------------------------------

import sounddevice as sd
import soundfile as sf
import librosa as lr
import numpy as np

# Load MP3 using librosa (soundfile cannot read mp3)
audio, sr_song = lr.load("voice1.wav", sr=None)

# Convert to mono if stereo
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Resample to 48k if needed
if sr_song != Fs:
    audio = lr.resample(audio, orig_sr=sr_song, target_sr=Fs)

# Normalize
audio = audio.astype(np.float32)
audio /= np.max(np.abs(audio)) + 1e-12

# Create a reverb instance for listening
rev_song = Reverb(
    max_delay_sec=1.0,
    a=0.7,        # warm damping
    g=0.4,        # smooth tail
    gAll=0.5,     # classic diffusion
    numb_Combs=6,
    numb_AllPass=3,
    room_Size=1,
    mix=0.7      # subtle reverb for music
)

# Process audio sample-by-sample
y_song = np.zeros_like(audio)
for n in range(len(audio)):
    y_song[n] = rev_song.process(audio[n])

# Normalize output
y_song /= np.max(np.abs(y_song)) + 1e-12

# PLAY IT
print("Playing song with reverb...")
sd.play(y_song, Fs)
sd.wait()

# OPTIONAL: Save output
sf.write("WhileYouDo_reverb.wav", y_song, Fs)
print("Saved WhileYouDo_reverb.wav")
