#include "DelayLine.hpp"

//Constructor
DelayLine::DelayLine(float delay_ms, float sample_rate){

    size = 98000;
    delay_samples = ((int)((sample_rate / 1000.0f) * delay_ms));
    for (int i=0 ; i < size ; i++){
        buffer[i]=0;
    }
    write_idx = 0;

}

float DelayLine::process(float x){

    int read = write_idx - delay_samples;
    if (read < 0){
        read += size;
    }
    //"read"
    float y = buffer[read];
    //write
    buffer[write_idx] = x;
    //move pointer
    write_idx ++;
    if (write_idx >= size){
        write_idx = 0;      
    }
    return y;

}



