#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 17:57:36 2026

@author: armandoiachini
"""
#------------------------------------------------
      #REVERB CLASSES
#------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf


#------------------------------------------------------------------------------
#   DELAY CLASS  
#------------------------------------------------------------------------------
class DelayLine:
    
    def __init__(self, max_delay_sec:int , delay_ms): #Arguments in __init__ are the things your class needs in order to be built
        self.Fs = 48000
        #Time delay
        self.delay_samples = int((delay_ms / 1000) * self.Fs)
        self.size = int(max_delay_sec  * self.Fs)
        # Allocate buffer
        self.buffer = np.zeros(self.size, dtype=np.float32)
        
        # Write index
        self.writeIndex = 0
        
    def process(self, x ):
        
        readIndex = (self.writeIndex - self.delay_samples) #% self.size
        
        if readIndex < 0:
            readIndex += self.size

        #Read sample delay
        delayed = self.buffer[readIndex]
        
        #Write
        self.buffer[self.writeIndex] = x
        
        #Move the pointer
        self.writeIndex += 1
        if self.writeIndex >= self.size:
            self.writeIndex = 0
            
        
        return delayed
    
    def read(self):
        readIndex = (self.writeIndex - self.delay_samples)
        if readIndex < 0:
            readIndex += self.size
        return self.buffer[readIndex]

    def write(self, x):
        self.buffer[self.writeIndex] = x
        self.writeIndex += 1
        if self.writeIndex >= self.size:
            self.writeIndex = 0
    
#------------------------------------------------------------------------------
 #One Pole Filter 
#------------------------------------------------------------------------------          

#y[n] = (1 - a) * x[n] + a * y[n-1]  ----> a= 0.9 = ---> (1 - 0.9) ---> 0.1 * input...
#(1 − a) = how much new signal enters the loop
#a = how much old signal stays in the loop

class OnePole:
    
    def __init__(self, a):

        if a > 1:
            a = 1
        if a < 0:
            a = 0
        self.a = a
        self.v_prev = 0.0
        
    def process(self, x):
        y = (1 - self.a) * x + self.a * self.v_prev
        self.v_prev = y
        return y
#------------------------------------------------------------------------------
#   LP COMB CLASS  
#------------------------------------------------------------------------------
#Higher g → longer tail  ----> feedback = g * LPF( d[n] )
# Higher a → stronger low‑pass  ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.9 → 0.1 * d[n])
# Lower a → weaker low‑pass    ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.1 → 0.9 * d[n])

class LPcomb:
    def __init__(self, max_delay_sec, delay_ms, a, g):
        self.delayLine = DelayLine(max_delay_sec, delay_ms)  
        self.onePole = OnePole(a)
        self.g = g
        self.feedback = 0.0
       

    def process(self, x):
        d = self.delayLine.process(x + self.feedback)
        filtered = self.onePole.process(d)     
        self.feedback = self.g * filtered       
        return filtered
    
#------------------------------------------------------------------------------
#   ALL PASS CLASS  
#------------------------------------------------------------------------------
#y [ n ] = (− g ⋅ x [ n ]  + x [ n − D ] + ( g ⋅ y [ n − D])   
class AllPass:
    
    def __init__(self, max_delay_sec, delay_ms, gAll):
        self.gAll = gAll
        self.delayLine = DelayLine(max_delay_sec, delay_ms)
        

    def process(self, x):
        d = self.delayLine.read()         # d = v[n-D]
        v = x + self.gAll * d             # v[n] = x[n] + g·v[n-D]
        self.delayLine.write(v)           # store v[n] for D samples from now
        y = d - self.gAll * v             # y[n] = v[n-D] - g·v[n]
        return y
#------------------------------------------------------------------------------
#   REVERBERATOR 
#------------------------------------------------------------------------------  
class Reverb:
    
    #Max 8 Comb -
    SMALL_ROOM_DELAYS  = [31, 37, 43, 53, 59, 67, 73, 83]
    MEDIUM_ROOM_DELAYS = [59, 67, 79, 89, 97, 107, 113, 127]
    LARGE_ROOM_DELAYS  = [97, 107, 113, 127, 137, 149, 157, 167]
    
    #Max 6 All Pass - prime numbers
    ALLPASS_DELAYS = [11, 13, 17, 19, 23, 29]

    def __init__(self, max_delay_sec,
                 a, g, gAll, numb_Combs, numb_AllPass,
                 room_Size, mix, predelay_ms):
        
        if room_Size == 0:
            delay_ms_list = self.SMALL_ROOM_DELAYS
        elif room_Size == 1:
            delay_ms_list = self.MEDIUM_ROOM_DELAYS
        elif room_Size == 2:
            delay_ms_list = self.LARGE_ROOM_DELAYS
        else:
            raise ValueError("Room Size must be 0 (small), 1 (medium), or 2 (large)")

        delay_ms_list = delay_ms_list[:numb_Combs]

        self.numb_Combs = numb_Combs
        self.combs = []
        self.allPass = []
        self.mix = mix
        self.preDelay = DelayLine(max_delay_sec, predelay_ms)

        for i in range(numb_Combs):
            comb = LPcomb(max_delay_sec, delay_ms_list[i], a, g)
            self.combs.append(comb)

        for i in range(numb_AllPass):
            ap = AllPass(max_delay_sec, self.ALLPASS_DELAYS[i], gAll)
            self.allPass.append(ap)
        
    def process(self, x):
        
        x_predelay = self.preDelay.process(x)

        comb_sum = 0.0   
        for comb in self.combs:
            comb_sum += comb.process(x_predelay)
            
        comb_sum /= self.numb_Combs

        yVerb = comb_sum
        for ap in self.allPass:
            yVerb = ap.process(yVerb)
            
        y = (1 - self.mix) * x + self.mix * yVerb

        return y