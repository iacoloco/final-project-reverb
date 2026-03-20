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

        
    def process(self, x):
        
        #Clamp if delay user is > then buffer size
        if self.delay_samples >= self.size:
            self.delay_samples = self.size - 1
        
        self.readIndex = self.writeIndex - self.delay_samples
        #or instead --->     self.readIndex = (self.writeIndex - self.delay_samples) % self.size and i dont need if self.readIndex < 0:

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
    
        
        
 #One Pole Filter
 #y[n] = (1 - a) * x[n] + a * y[n-1]
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
    
#
class LPcomb:
    
    def __init___(self, max_delay_sec, delay_ms, a , g):
        
        self.delayLine = DelayLine(max_delay_sec, delay_ms)
        self.OnePole = OnePole(a)
        self.d = 0
        self.g = g
        self.feedback = 0.0
        
        
    def process(self, x):
        
        d = self.delayLine.process(x + self.feedback)
        
        self.feedback = self.g * self.OnePole.process(d)
        
        return  d 

    
        
        
        
       
        
        

    
    
    
    
    
    
    