#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:51:44 2026

@author: armandoiachini
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LPComb Analysis
"""

import sys
import os

# Add parent folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverb_classes import LPcomb
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

# LPcomb(max_delay_sec, delay_ms, a, g)
lp = LPcomb(1, 50, 0.1, 0.5)
y = np.zeros(N)

for n in range(N):
    y[n] = lp.process(x[n])

t_ms = (np.arange(N) / Fs) * 1000

#Higher g → longer tail  ----> feedback = g * LPF( d[n] )
# Higher a → stronger low‑pass  ----> LPF(d[n])
# --- Full impulse response ---
plt.figure(figsize=(12,4))
plt.plot(t_ms, y)
plt.title("LPComb Impulse Response (delay = 50 ms, a = 0.1, g = 0.9)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 450)
plt.ylim(-1.2, 1.2)
plt.axvline(50, color='red', linestyle='--')
plt.show()


# --- Zoom: first 100 ms ---
plt.figure(figsize=(12,4))
plt.plot(t_ms, y)
plt.title("Zoom: First 100 ms")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(-1.2, 1.2)
plt.show()



# ---------------------------------------------------------
# 2) Spectrogram for LPComb with light damping (a = 0.1)
# ---------------------------------------------------------

x, sr = lr.load("IR48k.wav")
x = x.astype(np.float32)
x = x / np.max(np.abs(x))

#Higher g → longer tail  ----> feedback = g * LPF( d[n] )
# Higher a → stronger low‑pass  ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.9 → 0.1 * d[n])
# Lower a → weaker low‑pass    ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.1 → 0.9 * d[n])
# LPcomb(max_delay_sec, delay_ms, a, g)
lp1 = LPcomb(1, 50, 0.1, 0.9)

y1 = np.zeros_like(x)
for n in range(len(x)):
    y1[n] = lp1.process(x[n])

X1 = lr.stft(y1, n_fft=4096, hop_length=64)
X1_db = 20 * np.log10(np.abs(X1) + 1e-12)

time_axis = np.arange(X1_db.shape[1]) * 64 / sr
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
plt.title("LPComb Spectrogram (a = 0.1, light damping) (g = 0.9, Longh Tail)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 3)
plt.tight_layout()
plt.show()



# ---------------------------------------------------------
# 3) Spectrogram for LPComb with strong damping (a = 0.7)
# ---------------------------------------------------------


#Higher g → longer tail  ----> feedback = g * LPF( d[n] )
# Higher a → stronger low‑pass  ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.9 → 0.1 * d[n])
# Lower a → weaker low‑pass    ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.1 → 0.9 * d[n])
# LPcomb(max_delay_sec, delay_ms, a, g)
lp2 = LPcomb(1, 50, 0.9, 0.1)

y2 = np.zeros_like(x)
for n in range(len(x)):
    y2[n] = lp2.process(x[n])

X2 = lr.stft(y2, n_fft=4096, hop_length=64)
X2_db = 20 * np.log10(np.abs(X2) + 1e-12)

time_axis = np.arange(X2_db.shape[1]) * 64 / sr
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
plt.title("LPComb Spectrogram (a = 0.9, strong damping) (g = 0.1, Short Tail)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.xlim(0, 3)
plt.tight_layout()
plt.show()
