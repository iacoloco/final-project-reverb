//plugin.hpp
#pragma once
#include <rack.hpp>




using namespace rack;

// Declare the Plugin globally , defined in plugin.cpp
extern Plugin* pluginInstance;

// Declare each Model globally, defined in each module source file
extern rack::plugin::Model* modelMyOsc;
extern rack::plugin::Model* modelMyReverb;
