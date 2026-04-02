#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:10:15 2026

@author: armandoiachini
"""

# test_reverb.py

import numpy as np
import sounddevice as sd
import soundfile as sf
from reverb_classes import Reverb   # <--- import your class




x, Fs = sf.read("IR.wav", dtype="float32")
x = x / np.max(np.abs(x))

if x.ndim == 1:
    x = np.column_stack((x, x))

xL = x[:, 0]
xR = x[:, 1]

yL = np.zeros_like(xL)
yR = np.zeros_like(xR)

#----------------------------------------------------------------
reverbL = Reverb(
    max_delay_sec=1,
    a=0.75,        # strong damping
    g=0.70,        # moderate feedback
    gAll=0.78,     # strong diffusion
    numb_Combs=6,
    numb_AllPass=4,
    room_Size=0
)

reverbR = Reverb(
    max_delay_sec=1,
    a=0.75,
    g=0.70,
    gAll=0.80,     # slightly different for stereo
    numb_Combs=7,  # decorrelation
    numb_AllPass=4,
    room_Size=1
)




#    def __init__(self, max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass,  room_Size:int):
#reverbL = Reverb(1, 0.7, 0.75, 0.7, 6, 4, 1)
#reverbR = Reverb(1, 0.7, 0.75, 0.7, 4, 6, 1)



for n in range(len(xL)):
    yL[n] = reverbL.process(xL[n])
    yR[n] = reverbR.process(xR[n])

y = np.column_stack((yL, yR))

sd.play(x, Fs)
sd.wait()
sd.play(y, Fs)
sd.wait()
