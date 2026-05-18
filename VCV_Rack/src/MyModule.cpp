#include "plugin.hpp"
#include <cmath>


//Helper funtion to cut the edge of the square wave to avoid aliasing:

static inline float polyBLEP (float phase , float dt){
	if (dt <= 0.f)
        return 0.f;

	//right after the jump:  when the phase is right after 0
	if (phase < dt ){
		float x = phase / dt ; // makes the window dt goes 0 -> 0.5 -> 1 (little curve)
		//return the correction curve
		return 2*x - x*x - 1.f;                                 
	}
	//Right before the Top edge amplitude =1 ( phase - (1-dt))
	if (phase >  1.f -dt) {
		float x = (phase - 1.f) / dt; 
		return x*x + x+x + 1.f;
	}
	// Away from edges: no correction
    return 0.f;
	
}


struct MyOsc : Module {
    float phase = 0.f;
    float blinkPhase = 0.f;
	//Triangle current state:
	float triangleState = 0;

    
	enum ParamId {
		PITCH_PARAM,
		PARAMS_LEN
	};
	enum InputId {
		PITCH_INPUT,
		INPUTS_LEN
	};
	enum OutputId {
		SINE_OUTPUT,
		TRIANGLE_OUTPUT,
		SQUARE_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId {
		BLINK_LIGHT,
		LIGHTS_LEN
	};

	MyOsc() {
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		configParam(PITCH_PARAM, 0.f, 1.f, 0.f, "");
		configInput(PITCH_INPUT, "");
		configOutput(SINE_OUTPUT, "Sine");
		configOutput(TRIANGLE_OUTPUT,"Triangle");
		configOutput(SQUARE_OUTPUT, "Square");
	}

    // THE ACTUAL DSP CODE!!!!!!!!!!!!!!!!!!!!



    //This function run 48000 times x second meaning each sample is taken at 0.0000208 seconds (1/48000) 
    void process(const ProcessArgs &args) override {
        // Compute the frequency from the pitch parameter and input
        float pitch = params[PITCH_PARAM].getValue();
        pitch += inputs[PITCH_INPUT].getVoltage();
        // The default frequency is C4 = 261.6256 Hz
        float freq = dsp::FREQ_C4 * std::pow(2.f, pitch);
		

        // Accumulate the phase
        phase += freq * args.sampleTime;
        if (phase >= 1.f)
            phase -= 1.f;

  

		//Sine:
		float sine = std::sin(2.f * M_PI * phase);

		//Task 3: create a Square wave using polyBleep to avoid aliasing at high friquencies cuttig the edge of the square:
		//Regular funtions -> float square = (phase < 0.5f) ? 1.0f : -1.0f -> this will cause aliasing 
		//SQUARE:
		float dt = freq * args.sampleTime * 2.f;
		float square = (phase < 0.5f) ? 1.f : -1.f ;

		square += polyBLEP(phase , dt);

        float p2 = phase + 0.5f;
        p2 -= std::floor(p2);           // wrap to [0,1)

 
        square -= polyBLEP(p2, dt);

        const float leak = 0.9995f; // NEW: tiny decay factor to stop drift

        // NEW: update the triangle state (discrete-time leaky integrator)
        triangleState = leak * triangleState + (4.f * freq) * square * args.sampleTime;

        // NEW: clamp keeps it safe in [-1,1] range, avoids runaway values
        float triangle = clamp(triangleState, -1.f, 1.f);



        //OUTPUT:
		outputs[SINE_OUTPUT].setVoltage(5.f * sine);
		outputs[TRIANGLE_OUTPUT].setVoltage(5.f * triangle);
		outputs[SQUARE_OUTPUT].setVoltage(5.f * square);



        // Blink light at 1Hz
        blinkPhase += args.sampleTime;
        if (blinkPhase >= 1.f)
            blinkPhase -= 1.f;
        lights[BLINK_LIGHT].setBrightness(blinkPhase < 0.5f ? 1.f : 0.f);
    }

};

//GUI 

struct MyModuleWidget : ModuleWidget {
	MyModuleWidget(MyOsc* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/MyOsc.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(25.00, 10.00)), module, MyOsc::BLINK_LIGHT));

		addParam(createParamCentered<SynthTechAlco>(mm2px(Vec(25.00, 21.00)), module, MyOsc::PITCH_PARAM));

		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(25.00, 81.976)), module, MyOsc::PITCH_INPUT));

		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(12.50, 100.929)), module, MyOsc::SINE_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(25.00, 100.929)), module, MyOsc::TRIANGLE_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(38.00, 100.929)), module, MyOsc::SQUARE_OUTPUT));

		
	}
};


Model* modelMyOsc = createModel<MyOsc, MyModuleWidget>("MyOsc");
