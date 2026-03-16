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
    
    def __init__(self, D: int):
        if D < 1:
            raise ValueError("D need to be > then 1")

        
        self.D = int(D)
        self.buffer = np.zeros(self.D , dtype= np.float32)
        self.idx = 0
    
        # Flexible API for comb/allpass:
    def read(self) -> float:
        return float(self.buffer[self.idx])

    def write(self, x: float):
        self.buffer[self.idx] = float(x)

    #Move Pointer
    def movePointer(self):
        self.idx += 1
        if self.idx >= self.D:
           self.idx = 0
    
    
    # Pure delay: y[n] = x[n-D]
    def process_sample(self, x: float) ->float:
        
        #Read and output the bufferdelay
        y = float(self.buffer[self.idx])
        
        #Write the current input in to the buffer  
        self.buffer[self.idx] =float(x)
        
        #Move
        self.movePointer()
            
        return y
    
    def clear(self):
        self.idx=0
        self.buffer.fill(0.0)
    


    
   
class FeedbackComb:
    # y[n] = x[n] + g * y[n-D]
    def __init__(self, D: int, g: float):
        self.g = float(g)
        self.delay = DelayLine(D)

    def process_sample(self, x: float) -> float:
        yd = self.delay.read()                 # y[n-D]
        y  = float(x) + self.g * yd            # y[n]
        self.delay.write(y)                    # store y[n]
        self.delay.movePointer()               # advance delay line
        return y

    def clear(self):
        self.delay.clear()
        

   
    


class LPcomb:
    
    def __init__(self, D:int , g: float , damp: float ):
       
        if not (0.0 <= g < 1.0):
            raise ValueError("g must be in [0, 1)")

        if not (0.0 <= damp < 1.0):
            raise ValueError("damp must be in [0, 1)")
                
        self.D = int(D)
        self.g = float(g)
        self.damp = float(np.clip(damp, 0.0, 0.999999))
        self.delay = DelayLine(D)
        #One pole LPF state 
        self.v = 0.0
        
    def process_sample(self, x:float) -> float:
        
        #Read Delay Buffer
        yd = self.delay.read()
        
        #Compiute the one pole low pass
        self.v = (1 - self.damp) * yd + self.damp * self.v
        
        y = float(x) + self.g * self.v
        
        #Write the output on the delay buffer
        self.delay.write(y)
        
        #Move pointer
        self.delay.movePointer()
        
        return y
        
D = 4
dl = LPcomb(2, 0.5, 0.5)
x = [1,0,0,0,0,0,0]
y = [dl.process_sample(v) for v in x]
print(y)
   
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    