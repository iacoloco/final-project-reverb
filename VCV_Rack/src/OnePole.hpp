#ifndef ONEPOLE__HPP
#define ONEPOLE__HPP

struct OnePole{
    float a;
    float b;
    float y_prev;

    OnePole(float a);

    float process(float x);

    void setA(float new_a);



};

#endif