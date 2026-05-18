readme_text = """# 🎛️ MyReverb — Schroeder-Moorer Reverberator in Python and VCV Rack

A Schroeder-Moorer-style algorithmic reverberator built from scratch by **Armando Iachini** as part of a final-year Audio Software Engineering project at the University of West London.

The reverb was first designed, analysed, and validated as a Python prototype, then ported to real-time C++ as a VCV Rack plugin.

---

## 🧠 Project Overview

This project implements a **Schroeder-Moorer-style algorithmic reverberator**, a classic delay-based approach to artificial reverberation. The aim was to understand the design of an algorithmic reverb from first principles and then implement it in a real-time modular synthesis environment.

The main signal chain is:

Input → 8 Parallel Low-Pass Feedback Comb Filters → 4 Series All-Pass Filters → Wet/Dry Mix → Output

### Signal Chain Explained

- **Low-pass feedback comb filters** create the main reverberant tail. Each comb filter recirculates the signal through a delay line and includes a low-pass filter in the feedback loop to simulate high-frequency damping.

- **All-pass filters** increase echo density and diffusion, helping to smooth the reverb tail.

- **Wet/dry mix** blends the original input signal with the processed reverberated signal.

---

## 🏗️ Architecture

### Python Prototype: Design and Analysis

The reverb was first built in Python using object-oriented design, with one class for each DSP component:

| Class | Role |

|---|---|

| `DelayLine` | Circular buffer used to create delay-based effects |

| `OnePole` | One-pole low-pass filter: `y[n] = (1-a)x[n] + ay[n-1]` |

| `LPcomb` | Low-pass feedback comb filter using delay, feedback, and damping |

| `AllPass` | All-pass filter used to increase diffusion |

| `Reverb` | Top-level reverberator connecting all DSP blocks |

Each Python class was tested individually by processing an impulse response through its `process(x)` method. The outputs were plotted using Matplotlib and compared with the expected DSP behaviour before the complete reverb was assembled.

### C++ / VCV Rack: Real-Time Port

The Python prototype was then ported to C++ for implementation as a VCV Rack module. The VCV Rack version follows a per-sample processing structure using the `process()` method.

The implementation was designed with real-time audio constraints in mind, including predictable execution time and pre-allocated internal buffers.

---

## 📐 DSP Concepts Covered

- **Circular buffers** for delay-line implementation

- **Feedback comb filters** for creating the main reverberant decay

- **One-pole low-pass filters** for high-frequency damping

- **All-pass filters** for diffusion and echo-density increase

- **Freeverb-style tuning and parameter mapping** for practical reverb control

- **Real-time audio processing** inside VCV Rack

---

## 🔬 Validation and Analysis

The Python and VCV Rack implementations were tested using fixed parameter settings. The validation process included:

- **Impulse response analysis** to inspect the time-domain decay

- **Waveform comparison** between the Python and VCV Rack outputs

- **FFT magnitude analysis** to compare spectral behaviour

- **Energy Decay Curve (EDC) analysis** to evaluate how the reverberation tail decays over time

The results showed that the C++ VCV Rack implementation reproduced the main behaviour of the Python prototype. Small differences were observed in the spectral and decay analysis, which could be investigated further in future work.

---

## 🎛️ Parameters

| Parameter | Range | Description |

|---|---|---|

| `room_size` | 0.0 → 1.0 | Controls the feedback gain of the comb filters and therefore the perceived decay length |

| `damp` | 0.0 → 1.0 | Controls the one-pole low-pass coefficient and therefore the high-frequency damping |

| `wet` | 0.0 → 1.0 | Controls the amount of processed reverb signal in the output |

---

## ⏱️ Delay Times and Knob Mapping

The delay values were taken from the Freeverb source code on GitHub (Jezar, 2000) and converted into milliseconds.

A Freeverb-style mapping was adopted so that the `room_size` knob controls the feedback gain `g` of each comb filter, while the `damp` knob controls the one-pole low-pass coefficient `a`:

g = 0.28 * room_size + 0.7

a = 0.4 * damp

Python delay values:

delays_combs = [25.31, 26.94, 28.96, 30.75, 32.24, 33.81, 35.31, 36.67]

delay_Apf = [12.61, 10.00, 7.73, 5.10]

This mapping allows simple user-facing controls to adjust the internal DSP behaviour of the reverberator.

---

## 🛠️ Built With

- **Python 3**

- **NumPy**

- **SciPy**

- **Librosa**

- **Matplotlib**

- **Sounddevice**

- **C++17**

- **VCV Rack SDK v2**

- **Inkscape**

- **macOS Terminal and Make**

- **Git and GitHub**

---

## 📚 Learning Objectives

This project was developed as a final-year DSP and real-time audio software project. It demonstrates:

- Schroeder-Moorer reverberator design

- Delay-line-based DSP implementation

- Digital filter design using comb, all-pass, and one-pole filters

- Object-oriented audio programming in Python and C++

- Real-time audio constraints, including avoiding dynamic memory allocation in the audio thread

- VCV Rack plugin architecture, including parameters, inputs, outputs, and `process()`

- Audio validation using impulse responses, FFT analysis, waveform comparison, and decay curves

---

## 🚀 Future Work

Possible future improvements include:

- Stereo reverb implementation

- Further investigation of spectral and decay-time differences between Python and C++

- CPU performance profiling inside larger VCV Rack patches

- Improved graphical user interface

- Adaptive input-driven modulation

- Additional acoustic validation metrics

---

## 👤 Author

**Armando Iachini**  

BSc Audio Software Engineering  

University of West London  

📧 iachiniarmando@gmail.com  

> *Builted from scratch to understand every sample.*
