#include "LP_Comb.hpp"
#include "DelayLine.hpp"
#include "OnePole.hpp"


LP_Comb::LP_Comb(float delay_ms, float g, float a, float sample_rate)
    : delay(delay_ms, sample_rate), onepole(a)
{
    this->g = g;
    feedback = 0;
}


float LP_Comb::process( float x){

    float buffer = delay.process(x + feedback);

    float y = buffer;

    feedback = onepole.process(buffer) * g;
    
    return y;

}

void LP_Comb::setG( float new_g){
    g = new_g;
}
















