#ifndef ADAPTIVEDAMP
#define ADAPTIVEDAMP

struct Adaptive_damp{

    int idx;
    float buffer[480];
    float sum;
    int size;
    float average;
    int count=0;
    float sensitivity = 0;
    
    Adaptive_damp();

    bool process(float x);

};


#endif