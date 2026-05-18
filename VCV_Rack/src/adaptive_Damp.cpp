#include "adaptive_Damp.hpp"
#include "Reverb.hpp"
#include <cmath>
//Constructor 

 Adaptive_damp::Adaptive_damp(){

    idx= 0;
    size = 480;
    for(int i=0 ; i<size ; i++){
        buffer[i]=0;
    }
    float sum =0.0f;
    float average = 0.0f;
    int count =0;

}


bool Adaptive_damp::process(float x){
    float abs_x = std::fabs(x);
    // running 10 ms average
    sum -= buffer[idx];
    sum += abs_x;
    buffer[idx] = abs_x;

    idx +=1;
    if(idx >= size){
        idx = 0;
    }
    average = sum / size;

    // transient detected
    if( x > average * sensitivity ){
        count +=1;
        return true;

        if (count == 48000){
            count =0;
            average =0;
            sum=0;
            return false;
        }
    }
}