#ifndef ALLPASS_HPP
#define ALLPASS_HPP

struct AllPass{

    int delay_samples;
    int idx;
    float buffer[98000];
    float g_apf;
    int size;
    
    AllPass(float g_apf, float delay_ms, float sample_rate );

    float process( float x);

};


#endif

