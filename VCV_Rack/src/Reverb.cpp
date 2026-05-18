#include "Reverb.hpp"

Reverb::Reverb(float room_size , float damp, float wet, float sample_rate)
    :combs{
        //delays_combs =  [25.31, 26.94, 28.96, 30.75, 32.24, 33.81, 35.31, 36.67]
        //LP_Comb(float delay_ms, float g, float a, float sample_rate);
        LP_Comb( 25.31f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 26.94f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 28.96f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 30.75f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 32.24f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 33.81f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 35.31f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate),
        LP_Comb( 36.67f , room_size * 0.28f + 0.7f,  damp * 0.4f,  sample_rate)
    },

    all_passes{
        //delay_Apf = [12.61, 10.00, 7.73, 5.10]
        //AllPass(float g_apf, float delay_ms, float sample_rate );
        AllPass(0.5f, 12.61f, sample_rate),
        AllPass(0.5f, 10.00f, sample_rate),
        AllPass(0.5f, 7.73f, sample_rate),
        AllPass(0.5f, 5.10f, sample_rate)

    }
{
    this->wet = wet;

}

float Reverb::process(float x){

    float sum_comb = 0;
    for(int i=0; i < 8; i++){
        sum_comb += combs[i].process(x);
    }
    sum_comb = sum_comb / 6.0f;

    float apf1 = all_passes[0].process(sum_comb);
    float apf2 = all_passes[1].process(apf1);
    float apf3 = all_passes[2].process(apf2);
    float apf4 = all_passes[3].process(apf3);

    float mix = wet * apf4 + (1 - wet) * x;

    return mix;
}

void Reverb::setRoomSize(float new_room_size){
    float new_g = new_room_size * 0.28f + 0.7f;

    for (int i=0; i<8 ; i++){
        combs[i].setG(new_g);
    }   
}

void Reverb::setDamping(float new_damp){

    float new_a = new_damp * 0.4f;

    for( int i=0 ; i < 8 ; i++){
        combs[i].onepole.setA(new_a);
    }
}











 
    





