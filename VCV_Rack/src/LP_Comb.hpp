#ifndef LPCOMB_HPP
#define LPCOMB_HPP

#include "DelayLine.hpp"
#include "OnePole.hpp"


struct LP_Comb{

    float g;
    float feedback;
    DelayLine delay;
    OnePole onepole;

    LP_Comb(float delay_ms, float g, float a, float sample_rate);

    float process(float x);

    void setG(float new_g);



};

#endif