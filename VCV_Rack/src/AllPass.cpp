#include "AllPass.hpp"

//Constructor 

AllPass::AllPass(float g_apf, float delay_ms, float sample_rate){

    delay_samples = (int)((sample_rate / 1000.0f) * delay_ms);
    idx= 0;
    size = 98000;
    for(int i=0 ; i<size ; i++){
        buffer[i]=0;
    }
    this->g_apf = g_apf;

}


float AllPass::process(float x){

    float w = x + g_apf * buffer[idx];

    float y = -g_apf * w + buffer[idx];

    buffer[idx] = w;

    idx +=1;
    if(idx >= delay_samples){
        idx = 0;
    }

    return y;

}