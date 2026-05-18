#include "plugin.hpp"
#include "Reverb.hpp"
#include <cmath>

//Get sample rate from VCV Rack
//float sample_rate= APP->engine->getSampleRate();

struct MyReverb :Module {

	enum ParamId {
		ROOMSIZE_PARAM,
		DAMP_PARAM,
		WET_PARAM,
		PARAMS_LENS
	};

	enum InputId {
		AUDIO_INPUT,
        ROOMSIZE_CV_INPUT,
        DAMP_CV_INPUT,
        WET_CV_INPUT,
		INPUTS_LEN
	};

	enum OutputId {
		AUDIO_OUTPUT,
		OUTPUTS_LEN
	};

	enum lightId {
		LIGHTS_LEN
	};

    //Declere the reverb as a member 
    Reverb reverb;

    //Reverb::Reverb(float room_size , float damp, float wet, float sample_rate)
    //Constractor
	MyReverb() : reverb(0.5f, 0.5f, 0.3f, 48000.0f){



        config(PARAMS_LENS, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
        configParam(ROOMSIZE_PARAM, 0.f, 1.f, 0.5f, "Room size");
        configParam(DAMP_PARAM,     0.f, 1.f, 0.5f, "Damping");
        configParam(WET_PARAM,      0.f, 1.f, 0.3f, "Wet");
        configInput(AUDIO_INPUT, "Audio");
        configOutput(AUDIO_OUTPUT, "Audio");
        configInput(ROOMSIZE_CV_INPUT, "Decay CV");
        configInput(DAMP_CV_INPUT, "Damping CV");
        configInput(WET_CV_INPUT, "Wet CV");
    }

	void process(const ProcessArgs& args) override {
    float x = inputs[AUDIO_INPUT].getVoltage() / 5.0f;   // read
    float x_abs = std::fabs(x);

    //Get the values knobs
    float room_size_knob = params[ROOMSIZE_PARAM].getValue() ;
    float damp_knob = params[DAMP_PARAM].getValue();
    float wet_knob = params[WET_PARAM].getValue();

    //Get the voltages from inputs
    float room_size_cv_input = inputs[ROOMSIZE_CV_INPUT].getVoltage() / 10.0f;
    float damp_cv_input = inputs[DAMP_CV_INPUT].getVoltage() / 10.0f;
    float wet_cv_input = inputs[WET_CV_INPUT].getVoltage() / 10.0f;



    //Clamp
    float room_size = clamp(room_size_cv_input + room_size_knob, 0.0f , 1.0f);
    float damp = clamp(damp_cv_input + damp_knob, 0.0f ,1.0f);
    float wet = clamp(wet_cv_input + wet_knob, 0.0f ,1.0f);


    //Set the values on the reverb
    reverb.setRoomSize(room_size);
    //reverb.setDamping(x_abs + damp_knob );
    reverb.setDamping(damp);
    reverb.wet=(wet);
    

    float y = reverb.process(x);
    outputs[AUDIO_OUTPUT].setVoltage(y * 5.0f);         
	};

};


struct MyReverbWidget : ModuleWidget {
    MyReverbWidget(MyReverb* module) {
        setModule(module);
        setPanel(createPanel(asset::plugin(pluginInstance, "res/MyReverb.svg")));


        addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
        addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
        addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
        addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		//Knobs
        addParam(createParamCentered<SynthTechAlco>(mm2px(Vec(25.00, 37.00)), module, MyReverb::ROOMSIZE_PARAM));
        addParam(createParamCentered<SynthTechAlco>(mm2px(Vec(25.00, 60.00)), module, MyReverb::DAMP_PARAM));
        addParam(createParamCentered<SynthTechAlco>(mm2px(Vec(25.00, 85.00)), module, MyReverb::WET_PARAM));


		// Audio input and output jacks
        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(11.00, 115.929)), module, MyReverb::AUDIO_INPUT));
        addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(25.00, 115.929)), module, MyReverb::AUDIO_OUTPUT));
        //Input knobs
        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(11.0, 37.0)), module, MyReverb::ROOMSIZE_CV_INPUT));
        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(11.0, 60.0)), module, MyReverb::DAMP_CV_INPUT));
        addInput(createInputCentered<PJ301MPort>(mm2px(Vec(11.0, 85.0)), module, MyReverb::WET_CV_INPUT));
    }
};



Model* modelMyReverb = createModel<MyReverb, MyReverbWidget>("MyReverb");