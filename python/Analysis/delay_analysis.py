#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:27:43 2026

@author: armandoiachini
"""
import sys
import os

# Add parent folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverb_classes import DelayLine
import numpy as np
import matplotlib.pyplot as plt


Fs = 48000
N = 20000

def impulse(N):
    x = np.zeros(N, dtype=np.float32)
    x[0] = 1.0
    return x

x = impulse(N)
delay = DelayLine(max_delay_sec=1, delay_ms=50)
y = np.zeros(N)

for n in range(N):
    y[n] = delay.process(x[n])

t_ms = (np.arange(N) / Fs) * 1000

plt.figure(figsize=(12,4))
plt.plot(t_ms, y)
plt.title("DelayLine Impulse Response (50 ms)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.axvline(50, color='red', linestyle='--')
plt.show()

