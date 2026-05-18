#ifndef REVERB_HPP
#define REVERB_HPP

#include "LP_Comb.hpp"
#include "AllPass.hpp"

struct Reverb{

    float wet;

    LP_Comb combs[8];
    AllPass all_passes[4];

    Reverb(float room_size , float damp, float wet, float sample_rate);

    float process(float x);

    void setRoomSize(float new_room_size);
    void setDamping(float new_value);

};

#endif