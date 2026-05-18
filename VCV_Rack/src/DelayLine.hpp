#ifndef DELAYLINE_HPP
#define DELAYLINE_HPP

struct DelayLine {

    int delay_samples;
    float buffer[98000];
    int write_idx;
    int size;
    float a;

    DelayLine(float delay_ms , float sample_rate);

    float process(float x);


};

#endif