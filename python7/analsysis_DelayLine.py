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
import librosa as lr
import soundfile as sf  



Fs = 48000
N = Fs *6

x = np.zeros(N)
x[0]= 1 #delta
delayms=70
delay = DelayLine(delayms, Fs)

ir_DelayLine = np.zeros(N)
for i in range(N):
    ir_DelayLine[i] = delay.process(x[i])

# Time axis
t = np.arange(N) / Fs *1000

#------------------------------------------------------------------------------
" 1) Impulse Response  DELAY LINE CLASS"
#------------------------------------------------------------------------------

plt.figure()
plt.plot(t[:4000], ir_DelayLine[:4000])
plt.xlabel("Time (s)")
plt.xticks(np.arange(0, 80, 5))
plt.ylabel("Amplitude")
plt.title(f"Impulse Response delay at {delayms}ms")
plt.grid(True, alpha=0.5)
plt.show()

#------------------------------------------------------------------------------
" 2) Impulse Response  OnePole"
#------------------------------------------------------------------------------
a=0.1
onePole = OnePole(a)
ir_onePole = np.zeros(N)
for i in range(N):
    ir_onePole[i] = onePole.process(x[i])
t_ms = np.arange(N) / Fs *1000

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(t_ms[:20], ir_onePole[:20])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.title(f"Impulse Response One Pole     a={a}")
plt.grid(True)


#------------------------------------------------------------------------------
" 2.1) One Pole Magnetude Response - OnePole"
#------------------------------------------------------------------------------
FFT = np.fft.rfft(ir_onePole)
    
"Magnitude-----> abs: remove complex numbers"
magnitude = np.abs(FFT)

" convert bin number  Fs/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
#frequecies_Hz_axes = np.arange(0 , N/2 ) *freq_Values
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values

"Magnitude to db"
db = 20 * np.log10(magnitude + 1e-12)

plt.subplot(2, 1, 2)
plt.plot(frequecies_Hz_axes, db)
plt.xlabel("Hz")
plt.ylabel("db ")
plt.title(f"OnePole Magnetude Response.   a={a}")
plt.grid(True)
plt.subplots_adjust(hspace=0.4)
plt.show()


#------------------------------------------------------------------------------
" 3.0 IR LP Comb Class Impulsive response"
#------------------------------------------------------------------------------
#g ---> gain feedback
delay_ms = 35
g=0.4
LP_comb= LP_Comb(delay_ms, g, a, sample_rate = Fs)
ir_LP_Comb = np.zeros(N)

for i in range(N):
    ir_LP_Comb[i] = LP_comb.process(x[i])
    
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(t_ms[:12000] ,ir_LP_Comb[:12000])
plt.xlabel("time ms")
plt.ylabel("amplitude")
plt.title(f"IR LP Comb — g={g} ,Onepole a={a}, - delay= {delay_ms} ms")
plt.grid(True)


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

plt.subplot(2, 1, 2)
plt.plot(frequecies_Hz_axes[:3000], db_LP[:3000])
plt.xlabel("Hz")
plt.ylabel("db ")
plt.title(f"LP Comb Magnetude Response — g={g} Onepole a={a}  - delay= {delay_ms} ms")
plt.grid(True)
plt.subplots_adjust(hspace=0.4)
plt.show()

#------------------------------------------------------------------------------
" 4.0) IR Response - APF"
#------------------------------------------------------------------------------

delay_ms = 25
g_apf = 0.9

ir_apf = np.zeros(N)

all_pass = allPass(g_apf, delay_ms, Fs)

for i in range (N):
    ir_apf[i] = all_pass.process(x[i])
    
plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.plot(t_ms[:18000] , ir_apf[:18000])
plt.xlabel("Time ms")
plt.ylabel("Amplitude ")
plt.title(f"Impulse All Pass Filter - g_allPass={g_apf}, with delay={delay_ms} ms")
plt.grid(True)


#------------------------------------------------------------------------------
" 4.1) Frequency Response - APF"
#------------------------------------------------------------------------------

FFT_apf = np.fft.rfft(ir_apf)

"Magnitude-----> abs: remove complex numbers"
magnitude_apf = np.abs(FFT_apf)

" convert bin number  Fs/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
#frequecies_Hz_axes = np.arange(0 , N/2 ) *freq_Values
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values

"Magnitude to db"
db_apf = 20 * np.log10(magnitude_apf + 1e-12)

plt.subplot(2, 1, 2)
plt.plot(frequecies_Hz_axes, db_apf)
plt.xlabel("Hz")
plt.ylabel("db ")
plt.ylim(-1, 1)
plt.title(f"APF Magnetude Response — g_all={g_apf} ")
plt.grid(True)
plt.subplots_adjust(hspace=0.4)
plt.show()


#------------------------------------------------------------------------------
" 4.0) Python Reverb"
#------------------------------------------------------------------------------


"Load Clap file"
clap, sr_ir = lr.load("clap_48k.wav", sr=48000, mono=True)
Fs=48000

print("Sample rate: ", sr_ir)

# Add 4 seconds of silence at the end so reverb tail can fully decay
clap= np.concatenate([clap, np.zeros(Fs*8, dtype=clap.dtype)])

" Reverb parameters — change these to run a parameter sweep"
room_size = 0.5
damp      = 0.5
wet       = 1.0



reverb_python = Reverb(room_size=room_size, damp=damp, wet=wet, sample_rate=Fs)

"Process sample by sample"
y_reverb_python = np.zeros(len(clap), dtype=np.float32)
for i in range(len(clap)):
    y_reverb_python[i] = reverb_python.process(clap[i])
    

sf.write("python.wav", y_reverb_python, Fs, subtype='PCM_24')
print("Saved wet_python.wav")


"Time axis"
t_ir = np.arange(len(clap)) / Fs

"Plot dry vs wet Python"
plt.figure()
plt.plot(t_ir[:60000], clap[:60000],  label="Dry",  alpha=0.8)
plt.plot(t_ir[:60000], y_reverb_python[:60000], label="Wet", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Dry vs Wet Python room_s={room_size}, damp={damp}, wet={wet}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"Play dry then wet"
print("Playing DRY...")
sd.play(clap, Fs)
sd.wait()

print("Playing WET (medium room reverb)...")
sd.play(y_reverb_python, Fs)
sd.wait()


#------------------------------------------------------------------------------
" 4.1) Frequency Response - Reverb"
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
" Frequency Response - Python Reverb"
#------------------------------------------------------------------------------
"FFT of dry and wet signals"
FFT_python_output = np.fft.rfft(y_reverb_python)

"Magnitude → dB"

db_python_output = 20 * np.log10(np.abs(FFT_python_output) + 1e-12)

"Frequency axis"
freq_values = Fs / len(clap)
freq_axis = np.arange(0, len(clap)/2 + 1) * freq_values

"Plot Python Reverb output"
plt.figure()
plt.plot(freq_axis, db_python_output)
plt.xlabel("Hz")
plt.ylabel("dB")
plt.title(f"Frequency Response - Python Reverb f(room_s={room_size}, damp={damp}, wet={wet})")
plt.legend()
plt.grid(True)
plt.show()



#------------------------------------------------------------------------------
" 5) VCV RACK OUTPUT REVERB VS PYTHON"
#------------------------------------------------------------------------------

" 5.1 SIGNAL IN TIME "
#------------------------------------------------------------------------------

"Load Clap file"
vcv_reverb_output, sr_ir = lr.load("vcv_record_r0.5,d0.5,w1.wav", sr=48000, mono=False)
Fs=48000
vcv_reverb_output= vcv_reverb_output[0]


"Time axis"
t = np.arange(len(vcv_reverb_output)) / Fs

"Plot VCV REVERB OUTPUT"
plt.figure()
plt.plot(t[:5000], vcv_reverb_output[:5000],  label="VCV_Rack",  alpha=0.8)
plt.plot(t[:5000], y_reverb_python[:5000], label="Python", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"VCV Rack vs Python f(room_s={room_size}, damp={damp}, wet={wet},both)  ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"Play vcvt"
print("Playing DRY...")
sd.play(vcv_reverb_output, Fs)
sd.wait()




#------------------------------------------------------------------------------

" 5.2 FFT COMPARISON"
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
" 4) Furious Transform ----> overall frequency content of the whole recording"
#------------------------------------------------------------------------------
N = len(vcv_reverb_output)
"trimp python lengh to alight it with VCV Rack (shorter)"
y_reverb_python = y_reverb_python[:N]

"FFT"
FFT_python = np.fft.rfft(y_reverb_python)
FFT_vcv_rack = np.fft.rfft(vcv_reverb_output)
    
"Magnitude-----> abs: remove complex numbers"
magnitude_python = np.abs(FFT_python)
magnitude_vcv_rack = np.abs(FFT_vcv_rack)

" convert bin number  sr/2 (Nyquist)→ real Hz"
"Δf = sr / N"
freq_Values = Fs / N
frequecies_Hz_axes = np.arange(0 , N/2 +1) *freq_Values


plt.figure()
plt.plot(frequecies_Hz_axes[:10000],magnitude_vcv_rack[:10000] ,label="VCV_Rack", alpha=0.7)
plt.plot(frequecies_Hz_axes[:10000],magnitude_python[:10000],  label="Python", alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"VCV Rack vs Python")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlabel("Hz")
plt.ylabel("Magnitude")
plt.title("FFT — Linear magnitude: Python vs VCV Rack")
plt.show()


"In db"
magnitude_python_dB   = 20 * np.log10(magnitude_python   + 1e-12)
magnitude_vcv_rack_dB = 20 * np.log10(magnitude_vcv_rack + 1e-12)

plt.figure()
plt.semilogx(frequecies_Hz_axes, magnitude_vcv_rack_dB, label="VCV (C++)", alpha=0.7, linewidth=0.8)
plt.semilogx(frequecies_Hz_axes, magnitude_python_dB,   label="Python",    alpha=0.7, linewidth=0.8)
plt.xlabel("Hz")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f"FFT comparison – dB / log-frequency")
plt.show()



#------------------------------------------------------------------------------

" 5.3 SIGNALS ALIGNED FOR BETTER COMPARISON MEASURMENT ( REMOVING THE SILENT INITIAL PART  "
#------------------------------------------------------------------------------


threshold = 0.01

start_py  = np.argmax(np.abs(y_reverb_python)   > threshold)
start_vcv = np.argmax(np.abs(vcv_reverb_output) > threshold)
print(f"Python start: sample {start_py} ({start_py/Fs*1000:.1f} ms)")
print(f"VCV    start: sample {start_vcv} ({start_vcv/Fs*1000:.1f} ms)")
print(f"Time difference in ms = " , (start_py - start_vcv) / Fs *1000)

py_aligned  = y_reverb_python[start_py:]
vcv_aligned = vcv_reverb_output[start_vcv:]

N = min(len(py_aligned), len(vcv_aligned))
py_aligned  = py_aligned[:N]
vcv_aligned = vcv_aligned[:N]

t = np.arange(N) / Fs
plt.figure()
plt.plot(t[:50000], vcv_aligned[:50000], label="VCV_Rack", alpha=0.7)
plt.plot(t[:50000], py_aligned[:50000],  label="Python",   alpha=0.7)
plt.title("VCV Rack vs Python Aligned")
plt.legend()
plt.grid(True)


#------------------------------------------------------------------------------
" 5.4 EDC DIAGNOSTIC PLOT"
#------------------------------------------------------------------------------

" Reload Python from file so we have the FULL untrimmed version"
y_python_reverb, _ = lr.load("python.wav", sr=Fs, mono=True)

def make_edc(signal):
    energy = signal ** 2                              # h²(s) -- squared impulse response
    edc    = np.cumsum(energy[::-1])[::-1]            # d(t) = integral from t to t_K of h²(s) ds
    edc    = edc / np.max(edc)                        # normalize so d(0) = 1 (i.e., 0 dB at start)
    return 10 * np.log10(edc + 1e-12)                 # convert to dB

edc_py  = make_edc(y_python_reverb)
edc_vcv = make_edc(vcv_reverb_output)

t_py  = np.arange(len(edc_py))  / Fs
t_vcv = np.arange(len(edc_vcv)) / Fs

plt.figure()
plt.plot(t_py,  edc_py,  label="Python",     alpha=0.8)
plt.plot(t_vcv, edc_vcv, label="VCV (C++)",  alpha=0.8)

plt.axhline(-60, color='gray', linestyle='--', linewidth=0.8, label="-60 dB")
plt.xlabel("Time (s)")
plt.ylabel("Energy (dB)")
plt.title(f"Energy Decay Curves - {suffix} ")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_vcv[48000:98000], edc_vcv[48000:98000], label="VCV (C++)",  alpha=0.8)
plt.plot(t_py[48000:98000],  edc_py[48000:98000],  label="Python",     alpha=0.8)


plt.axhline(-60, color='gray', linestyle='--', linewidth=0.8, label="-60 dB")
plt.xlabel("Time (s)")
plt.ylabel("Energy (dB)")
plt.title(f"Energy Decay Curves {suffix} ")
plt.legend()
plt.grid(True)
plt.show()