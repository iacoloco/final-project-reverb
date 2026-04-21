#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:22:38 2026

@author: armandoiachini
"""

from reverb import DelayLine
from reverb import OnePole
from reverb import LP_Comb
from reverb import allPass
from reverb import Reverb
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

Fs = 48000
N = Fs *6

x = np.zeros(N)
x[0]= 1 #delta

ir = np.zeros(N)

delay = DelayLine(100, Fs)


ir_DelayLine = np.zeros(N)
for i in range(N):
    ir_DelayLine[i] = delay.process(x[i])
    

# Time axis
t = np.arange(N) / Fs 

#------------------------------------------------------------------------------
" 1) Impulse Response  DELAY LINE CLASS"
#------------------------------------------------------------------------------
plt.figure()
plt.plot(t[:18000], ir_DelayLine[:18000])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Impulse Response")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 2) Impulse Response  OnePole"
#------------------------------------------------------------------------------
a=0.4
onePole = OnePole(a)
ir_onePole = np.zeros(N)
for i in range(N):
    ir_onePole[i] = onePole.process(x[i])
t_ms = np.arange(N) / Fs * 1000   

plt.figure()
plt.plot(t_ms[:20], ir_onePole[:20])
plt.xlabel("Time ms")
plt.ylabel("Amplitude ")
plt.title(f"Impulse Response One Pole - a={a}")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 2.1) One Pole Magnetude Response - OnePole"
#------------------------------------------------------------------------------
FFT = np.fft.rfft(ir_onePole)
    
"Magnitude-----> abs: remove complex numbers"
magnitude = np.abs(FFT)

" convert bin number  Fs/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values

"Magnitude to db"
db = 20 * np.log10(magnitude + 1e-12)

plt.figure()
plt.plot(frequecies_Hz_axes, db)
plt.xlabel("Hz")
plt.ylabel("db ")
plt.title(f"OnePole Magnetude Response — a={a}")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 3.0 IR LP Comb Class Impulsive response"
#------------------------------------------------------------------------------
#g ---> gain feedback
delay_ms = 50
g=0.9
LP_comb= LP_Comb(delay_ms, g, a, sample_rate = Fs)
ir_LP_Comb = np.zeros(N)

for i in range(N):
    ir_LP_Comb[i] = LP_comb.process(x[i])
    
plt.figure()
plt.plot(t_ms[:12000] ,ir_LP_Comb[:12000])
plt.xlabel("time ms")
plt.ylabel("amplitude")
plt.title(f"IR LP Comb — g={g} ,Onepole a={a}")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 3.1) Frequency Response - LP Comb"
#------------------------------------------------------------------------------
FFT_comb = np.fft.rfft(ir_LP_Comb)
    
"Magnitude-----> abs: remove complex numbers"
magnitude_comb = np.abs(FFT_comb)

" convert bin number  Fs/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values

"Magnitude to db"
db_LP = 20 * np.log10(magnitude_comb + 1e-12)

plt.figure()
plt.plot(frequecies_Hz_axes[:1000], db_LP[:1000])
plt.xlabel("Hz")
plt.ylabel("db ")
plt.title(f"LP Comb Magnetude Response — g={g} Onepole a={a} ")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 4.0) IR Response - APF"
#------------------------------------------------------------------------------

delay_ms = 13 
g_apf = 0.5

ir_apf = np.zeros(N)

all_pass = allPass(g_apf, delay_ms, Fs)

for i in range (N):
    ir_apf[i] = all_pass.process(x[i])
    
plt.figure()
plt.plot(t_ms[:10000] , ir_apf[:10000])
plt.xlabel("Time ms")
plt.ylabel("Amplitude ")
plt.title(f"Impulse All Pass Filter - g_allPass={g_apf}")
plt.grid(True)
plt.show()

#------------------------------------------------------------------------------
" 4.1) Frequency Response - APF"
#------------------------------------------------------------------------------

FFT_apf = np.fft.rfft(ir_apf)

"Magnitude-----> abs: remove complex numbers"
magnitude_apf = np.abs(FFT_apf)

" convert bin number  Fs/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values

"Magnitude to db"
db_apf = 20 * np.log10(magnitude_apf + 1e-12)

plt.figure()
plt.plot(frequecies_Hz_axes, db_apf)
plt.xlabel("Hz")
plt.ylabel("db ")
plt.ylim(-1, 1)
plt.title(f"APF Magnetude Response — g_all={g_apf} ")
plt.grid(True)
plt.show()


#------------------------------------------------------------------------------
" 4.0) IR Response - Reverb"
#------------------------------------------------------------------------------

import librosa as lr

"Load IR wav file"
x_ir, sr_ir = lr.load("clap.wav", sr=48000, mono=True)
Fs=48000

print("Sample rate: ", sr_ir)
print("Duration:    ", len(x_ir) / sr_ir, "s")

# Add 4 seconds of silence at the end so reverb tail can fully decay
x_ir = np.concatenate([x_ir, np.zeros(Fs*3, dtype=x_ir.dtype)])
"Create reverb"
reverb_test = Reverb(room_size=0.1, dump=1, wet=0.5, sample_rate=Fs)

"Process sample by sample"
y_reverb = np.zeros(len(x_ir), dtype=np.float32)
for i in range(len(x_ir)):
    y_reverb[i] = reverb_test.process(x_ir[i])


"Time axis"
t_ir = np.arange(len(x_ir)) / Fs

"Plot dry vs wet"
plt.figure()
plt.plot(t_ir, x_ir,  label="Dry",  alpha=0.8)
plt.plot(t_ir, y_reverb, label="Wet", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("IR.wav - Dry vs Wet")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"Play dry then wet"
print("Playing DRY...")
sd.play(x_ir, Fs)
sd.wait()

print("Playing WET (medium room reverb)...")
sd.play(y_reverb, Fs)
sd.wait()


#------------------------------------------------------------------------------
" 4.1) Frequency Response - Reverb"
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
" Frequency Response - Reverb (Dry vs Wet)"
#------------------------------------------------------------------------------
"FFT of dry and wet signals"
FFT_dry = np.fft.rfft(x_ir)
FFT_wet = np.fft.rfft(y_reverb)

"Magnitude → dB"
db_dry = 20 * np.log10(np.abs(FFT_dry) + 1e-12)
db_wet = 20 * np.log10(np.abs(FFT_wet) + 1e-12)

"Frequency axis"
freq_values = Fs / len(x_ir)
freq_axis = np.arange(0, len(x_ir)/2 + 1) * freq_values

"Plot Dry"
plt.figure()
plt.plot(freq_axis, db_dry, label="Dry", alpha=0.8)
plt.xlabel("Hz")
plt.ylabel("dB")
plt.title(f"Frequency Response - Original Signal decay={0.5} wet={0.5}")
plt.legend()
plt.grid(True)
plt.show()


"Plot Wet"
plt.figure()
plt.plot(freq_axis, db_wet, label="Wet", alpha=0.8)
plt.xlabel("Hz")
plt.ylabel("dB")
plt.title(f"Frequency Response - Reverb Wet decay={0.5} wet={0.5}")
plt.legend()
plt.grid(True)
plt.show()




