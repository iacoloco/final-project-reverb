"""
Reverb Analysis Script
- Impulse Response
- Frequency Response (FFT)
- Spectrogram (STFT)
- RT60 Decay
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reverb_classes import Reverb
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from scipy.signal import medfilt

 
#------------------------------------------------------------------------------
" 1) Generate Impulse "
#------------------------------------------------------------------------------
Fs = 48000
duration_sec = 6
N = Fs * duration_sec
 
impulse = np.zeros(N, dtype=np.float32)
impulse[0] = 1.0
 
#------------------------------------------------------------------------------
" 2) Create Reverbs - One per Room Size "
#------------------------------------------------------------------------------
# Parameters shared across all rooms
a          = 0.9     # damping (OnePole)
g          = 0.85    # feedback strength (longer tail)
gAll       = 0.5     # allpass feedback
numb_Combs    = 8
numb_AllPass  = 6
mix           = 1.0  # --->full wet 
predelay_ms   = 20.0
max_delay_sec = 1.0
 
reverb_small  = Reverb(max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass, 0, mix, predelay_ms)
reverb_medium = Reverb(max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass, 1, mix, predelay_ms)
reverb_large  = Reverb(max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass, 2, mix, predelay_ms)
 
#------------------------------------------------------------------------------
" 3) Process Impulse Through Each Reverb "
#------------------------------------------------------------------------------
ir_small  = np.zeros(N, dtype=np.float32)
ir_medium = np.zeros(N, dtype=np.float32)
ir_large  = np.zeros(N, dtype=np.float32)

for i in range(N):
    ir_small[i] = reverb_small.process(impulse[i])
    ir_medium[i] = reverb_medium.process(impulse[i])
    ir_large[i] = reverb_large.process(impulse[i])

# Time axis
t = np.arange(N) / Fs


#------------------------------------------------------------------------------
" 4) Impulse Response - Time Domain "
#------------------------------------------------------------------------------
plt.figure()
plt.plot(t[:48000], ir_small[:48000])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Impulse Response - Small Room")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t[:48000], ir_medium[:48000], color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Impulse Response - Medium Room")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t[:48000], ir_large[:48000], color="green")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Impulse Response - Large Room")
plt.grid(True)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------
" 4.1) Impulse Response - Time Domain "
#------------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t[:48000], ir_small[:48000])
axes[0].set_title("Impulse Response - Small Room")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(t[:48000], ir_medium[:48000], color="orange")
axes[1].set_title("Impulse Response - Medium Room")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

axes[2].plot(t[:48000], ir_large[:48000], color="green")
axes[2].set_title("Impulse Response - Large Room")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (s)")
axes[2].grid(True)

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
" 5) Frequency Response - FFT "
#------------------------------------------------------------------------------


FFT_small  = np.fft.rfft(ir_small)

"Magnitude -----> abs: remove complex numbers"
magnitude_small  = np.abs(FFT_small)


"Convert bin number to real Hz"
"Δf = Fs / N"
freq_Values    = Fs / N
frequencies_Hz = np.arange(0, N / 2 + 1) * freq_Values

"--- Plot 1 - Linear Magnitude - shows comb resonances ---"
plt.figure()
plt.plot(frequencies_Hz, magnitude_small,  label="Small",  alpha=0.8)

plt.xlabel("Hz")
plt.ylabel("Magnitude")
plt.title("Frequency Response ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"Convert to dB"
dB_small  = 20 * np.log10(magnitude_small  + 1e-8)



"--- Plot 2 - dB - ---"
plt.figure()
plt.plot(frequencies_Hz, dB_small,  label="Small",  alpha=0.8)
plt.xlabel("Hz")
plt.ylabel("Magnitude (dB)")
plt.title("Frequency Response db")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------
" 6) Spectrogram - STFT "
#------------------------------------------------------------------------------
hop_length = 512

"STFT of each impulse response"
Xstft_small  = lr.stft(ir_small,  hop_length=hop_length)
Xstft_medium = lr.stft(ir_medium, hop_length=hop_length)
Xstft_large  = lr.stft(ir_large,  hop_length=hop_length)
"Xstft.shape [freq_bins : frames]"

"Log Magnitude"
log_mag_small  = np.log(np.abs(Xstft_small)  + 1e-8)
log_mag_medium = np.log(np.abs(Xstft_medium) + 1e-8)
log_mag_large  = np.log(np.abs(Xstft_large)  + 1e-8)

"Define frequency axis in Hz"
total_bins    = log_mag_small.shape[0]
freq_Hz_axesY = np.linspace(0, Fs / 2, total_bins)

"Define time axis"
n_frames_stft  = log_mag_small.shape[1]
time_axis_stft = np.arange(n_frames_stft) * hop_length / Fs

"Plot Small"
plt.figure()
plt.imshow(
    log_mag_small,
    aspect='auto',
    origin='lower',
    extent=[time_axis_stft[0], time_axis_stft[-1], freq_Hz_axesY[0], freq_Hz_axesY[-1]]
)
plt.xlabel("Time (s)")
plt.ylabel("Hz")
plt.title("Spectrogram - Small Room")
plt.colorbar(label="Log Magnitude")
plt.tight_layout()
plt.show()

"Plot Medium"
plt.figure()
plt.imshow(
    log_mag_medium,
    aspect='auto',
    origin='lower',
    extent=[time_axis_stft[0], time_axis_stft[-1], freq_Hz_axesY[0], freq_Hz_axesY[-1]]
)
plt.xlabel("Time (s)")
plt.ylabel("Hz")
plt.title("Spectrogram - Medium Room")
plt.colorbar(label="Log Magnitude")
plt.tight_layout()
plt.show()

"Plot Large"
plt.figure()
plt.imshow(
    log_mag_large,
    aspect='auto',
    origin='lower',
    extent=[time_axis_stft[0], time_axis_stft[-1], freq_Hz_axesY[0], freq_Hz_axesY[-1]]
)
plt.xlabel("Time (s)")
plt.ylabel("Hz")
plt.title("Spectrogram - Large Room")
plt.colorbar(label="Log Magnitude")
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
" 7) RT60 - Energy Decay "
# RT60 = time for the reverb tail to decay by 60dB
#------------------------------------------------------------------------------



def compute_RT60(ir, Fs):
    " Convert amplitude to energy "
    energy = ir ** 2

    " ---- Current approach: Median filter ---- "
    energy_smooth = medfilt(energy, kernel_size=11)

    " Find peak energy "
    peak_energy = np.max(energy_smooth)

    " Threshold = -60dB below peak "
    threshold = peak_energy * (10 ** (-60 / 10))

    " Find last sample above threshold "
    above_threshold = np.where(energy_smooth > threshold)[0]

    if len(above_threshold) == 0:
        return 0.0

    last_sample = above_threshold[-1]
    rt60 = last_sample / Fs
    return rt60

rt60_small  = compute_RT60(ir_small,  Fs)
rt60_medium = compute_RT60(ir_medium, Fs)
rt60_large  = compute_RT60(ir_large,  Fs)

print("RT60 Small  Room: ", round(rt60_small), "s")
print("RT60 Medium Room: ", round(rt60_medium, 3), "s")
print("RT60 Large  Room: ", round(rt60_large,  3), "s")

" Last sample above threshold - to check if buffer is long enough "
print("Last sample Small:  ", round(rt60_small  * Fs), "of", N, "samples")
print("Last sample Medium: ", round(rt60_medium * Fs), "of", N, "samples")
print("Last sample Large:  ", round(rt60_large  * Fs), "of", N, "samples")

#------------------------------------------------------------------------------
" 8) Test with IR.wav file "
#------------------------------------------------------------------------------

import librosa as lr

"Load IR wav file"
x_ir, sr_ir = lr.load("IR.wav", sr=48000, mono=True)
"lr.load automatically: resamples to 48000, converts to float32, normalises to -1/1"

print("Sample rate: ", sr_ir)
print("Duration:    ", len(x_ir) / sr_ir, "s")

"Create reverb"
reverb_test = Reverb(max_delay_sec, a, g, gAll, numb_Combs, numb_AllPass, 1, mix, predelay_ms)

"Process sample by sample"
y = np.zeros(len(x_ir), dtype=np.float32)
for i in range(len(x_ir)):
    y[i] = reverb_test.process(x_ir[i])


"Time axis"
t_ir = np.arange(len(x_ir)) / Fs

"Plot dry vs wet"
plt.figure()
plt.plot(t_ir, x_ir,  label="Dry",  alpha=0.8)
plt.plot(t_ir, y, label="Wet", alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("IR.wav - Dry vs Wet (Medium Room)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"Play dry then wet"
print("Playing DRY...")
sd.play(x_ir, Fs)
sd.wait()

print("Playing WET (medium room reverb)...")
sd.play(y, Fs)
sd.wait()