#include "OnePole.hpp"

//Constructor
OnePole::OnePole(float a){
    // this = this obj
    // ->a = reach the field a 
    this->a=a;
    b = 1 - a; 
    y_prev = 0;
}

float OnePole::process(float x){
    // JOS: y (n) = b*x(n) - a*y(n- 1).---> a is negative (-a)
    float y = x * b + a * y_prev;
    y_prev = y;
    return y;

}

void OnePole::setA(float new_a){
    a = new_a;
    b = 1 - new_a;
}


