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
        self.readIndex=0
        
    def process(self, x ):
        
        self.readIndex = (self.writeIndex - self.delay_samples) #% self.size
        
        if self.readIndex < 0:
            self.readIndex += self.size

        #Read sample delay
        delayed = self.buffer[self.readIndex]
        
        #Write
        self.buffer[self.writeIndex] = x
        
        #Move the pointer
        self.writeIndex += 1
        if self.writeIndex >= self.size:
            self.writeIndex = 0
            
        
        return delayed
    
#------------------------------------------------------------------------------
 #One Pole Filter 
#------------------------------------------------------------------------------          

#y[n] = (1 - a) * x[n] + a * y[n-1]  ----> a= 0.9 = ---> (1 - 0.9) ---> 0.1 * input...
#(1 − a) = how much new signal enters the loop
#a = how much old signal stays in the loop

class OnePole:
    
    def __init__(self, a ):
        
        self.a = a
        self.v_prev= 0.0
        
    def process(self, x):
        if self.a > 1:
            self.a = 1
        if self.a < 0:
            self.a = 0
        
        y = (1 - self.a) * x + self.a * self.v_prev
        
        self.v_prev= y
        
        return y

#------------------------------------------------------------------------------
#   LP COMB CLASS  
#------------------------------------------------------------------------------
#Higher g → longer tail  ----> feedback = g * LPF( d[n] )
# Higher a → stronger low‑pass  ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.9 → 0.1 * d[n])
# Lower a → weaker low‑pass    ----> LPF(d[n]) = (1 - a) * d[n]  (a = 0.1 → 0.9 * d[n])

class LPcomb:
                                                           # g----->Feedback Strength (how long)
    def __init__(self, max_delay_sec, delay_ms, a , g):   # a---> dump (Bright vs warm) ---> 
        
        self.delayLine = DelayLine(max_delay_sec, delay_ms)
        self.onePole = OnePole(a)
        self.d = 0
        self.g = g
        self.feedback = 0.0
        
        
    def process(self, x):
        
        d = self.delayLine.process(x + self.feedback)
        
        self.feedback = self.g * self.onePole.process(d)
        
        return  d 
    
#------------------------------------------------------------------------------
#   ALL PASS CLASS  
#------------------------------------------------------------------------------
#y [ n ] = (− g ⋅ x [ n ]  + x [ n − D ] + ( g ⋅ y [ n − D])   
class AllPass:
    
    def __init__(self, max_delay_sec, delay_ms, gAll):
        self.gAll = gAll
        self.delayLine = DelayLine(max_delay_sec, delay_ms)
        self.feedback = 0.0   # stores y[n - D]
        
    def process(self, x):
        # Feed input + feedback into delay line
        d = self.delayLine.process(x + self.gAll * self.feedback)
        
        # AllPass output
        y = d - self.gAll * x
        
        # Store for next sample 
        self.feedback = y
        
        return y


#------------------------------------------------------------------------------
#   REVERBERATOR 
#------------------------------------------------------------------------------  
class Reverb:
    
    #Max 8 Comb
    SMALL_ROOM_DELAYS = [20, 24, 28, 32, 36, 40, 44, 48]
    MEDIUM_ROOM_DELAYS = [40, 46, 52, 58, 64, 70, 76, 82]
    LARGE_ROOM_DELAYS = [60, 68, 76, 84, 92, 100, 108, 116]
    
    #Max 6 All Pass
    ALLPASS_DELAYS = [10, 12, 14, 16, 18, 20]

    def __init__(self, max_delay_sec,
                 a, g, gAll, numb_Combs, numb_AllPass,
                 room_Size:int, mix:float):
        
        # Select delay list based on room size
        if room_Size == 0:
            delay_ms_list = self.SMALL_ROOM_DELAYS
        elif room_Size == 1:
            delay_ms_list = self.MEDIUM_ROOM_DELAYS
        elif room_Size == 2:
            delay_ms_list = self.LARGE_ROOM_DELAYS
        else:
            raise ValueError("Room Size must be 0 (small), 1 (medium), or 2 (large)")

        # Slice list based on number of combs
        delay_ms_list = delay_ms_list[:numb_Combs]

        # Store parameters
        self.numb_Combs = numb_Combs
        #List of Object of Combs
        self.combs = []
        #List of OBject of All Pass Filter
        self.allPass = []
        self.a = a
        self.g = g
        self.gAll = gAll
        self.mix = mix

        # Create N comb filters
        for i in range(numb_Combs):
            comb = LPcomb(max_delay_sec, delay_ms_list[i], a, g)
            self.combs.append(comb)

        # One allpass
        for i in range(numb_AllPass):
            ap = AllPass(max_delay_sec, self.ALLPASS_DELAYS[i], gAll)
            self.allPass.append(ap)
        

    def process(self, x):

        # Run combs in (Parallel)
        comb_sum = 0.0   
        for comb in self.combs:
            comb_sum += comb.process(x)
            
        #Normalize by number of combs
        comb_sum /= self.numb_Combs

        # Run through allpass (Series)
        yVerb = comb_sum
        for ap in self.allPass:
            yVerb = ap.process(yVerb)
            
        y = (1 - self.mix) * x + self.mix * yVerb
            

        return y

        
        