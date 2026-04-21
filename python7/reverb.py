
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:21:19 2026

@author: armandoiachini
"""

import numpy as np


class DelayLine:
    

    def __init__(self, delay_ms,  sample_rate):
        
        self.Fs = sample_rate
        self.delay_samples = int((self.Fs / 1000) * delay_ms)
        self.size = self.Fs *2
        self.buffer = np.zeros(self.size)
        self.write_idx = 0
        
    def process(self, x):
        
        read_idx = self.write_idx - self.delay_samples
        if read_idx < 0:
            read_idx += self.size
        "read"
        y = self.buffer[read_idx]
        
        "write"
        self.buffer[self.write_idx] = x
        "Move pointer"
        self.write_idx +=1
        if self.write_idx >= self.size:
            self.write_idx =0
            
        return y
        


class OnePole:

    def __init__(self, a):

        self.a = a
        self.b = 1-self.a
        self.y_prev = 0
 
        #dsp--> y(n) a * x(n) + b * y(n-1)
    def process(self, x):
        
        #y = (1 - self.a) * x + self.y_prev * self.a
        y = self.b*x + self.a * self.y_prev 
        self.y_prev = y
        
        return y
    
        
#In LPcomb, what came out of the delay line is what you feed into OnePole
class LP_Comb:
    
     def __init__(self, delay_ms, g, a, sample_rate):
        
        self.g = g
        self.delay = DelayLine(delay_ms, sample_rate)
        self.OnePole = OnePole(a)
        self.feedback = 0
        
     def process(self, x):
        
        combined = x + self.feedback
        d = self.delay.process(combined)
        y = d
        self.feedback = self.OnePole.process(d)
        self.feedback = self.feedback * self.g
        return y
        


class allPass:

    
    def __init__(self, g_apf, delay_ms, sample_rate):
        
        self.Fs = sample_rate
        self.delay_ms = delay_ms
        self.delay_samples = int((self.Fs / 1000) * self.delay_ms)
        self.buffer = np.zeros(self.delay_samples)
        self.g_apf = g_apf
        self.idx = 0
        
    def process(self, x):
        
        w = x + self.g_apf * self.buffer[self.idx]
        
        y = -self.g_apf * w + self.buffer[self.idx]
        
        self.buffer[self.idx] = w 
        
        self.idx +=1
        if self.idx >= len(self.buffer) :
            self.idx = 0
            
        return y
            
        
        
        
class Reverb:
    
    def __init__(self, room_size , dump ,wet ,sample_rate):
        
        self.Fs= sample_rate
        self.g = room_size * 0.28 + 0.7
        self.wet = wet
        self.a= dump * 0.4
        g_apf=0.5
        delays_combs =  [25.31, 26.94, 28.96, 30.75, 32.24, 33.81, 35.31, 36.67]
        delay_Apf = [12.61, 10.00, 7.73, 5.10] 
        
        self.combs = []
        self.All_pass = []
        
       
        for i in range(8): 
            #     def __init__(self, delay_ms, g, a, sample_rate):
            comb = LP_Comb(delays_combs[i], self.g, self.a ,self.Fs)
            self.combs.append(comb)
            
            
        for i in range(4):
            apf = allPass(g_apf, delay_Apf[i] , self.Fs)
            self.All_pass.append(apf)
    
    
    
    def process(self, x):
        # Step 1: run x through all combs, sum output

        
        sum_comb = 0
        for comb in self.combs:
            sum_comb += comb.process(x)
        sum_comb = sum_comb / 8
            
         # Step 2: pass sum through allpasses in series
        apf1 = self.All_pass[0].process(sum_comb)
        
        apf2 = self.All_pass[1].process(apf1)
        
        apf3 = self.All_pass[2].process(apf2)
        
        apf4 = self.All_pass[3].process(apf3)
        
        
        # Step 3: mix wet and dry
        y = self.wet * apf4 + (1 - self.wet) * x
         
        return y
        
        
        
  

    
        
        

    

    

    
    
    
    
    
    
