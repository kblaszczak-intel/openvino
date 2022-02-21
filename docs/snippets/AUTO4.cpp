#include <openvino/openvino.hpp>

int main() {
	ov::Core core;

	// Read a network in IR, PaddlePaddle, or ONNX format:
	std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
//! [part4]
	// Example 1
	// Compile and load networks:
	ov::CompiledModel compiled_model0 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "HIGH"}});
	ov::CompiledModel compiled_model1 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "MEDIUM"}});
	ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "LOW"}});
	/************
	  Assume that all the devices (CPU, GPU, and MYRIAD) can support all the networks.
   	  Result: compiled_model0 will use GPU, compiled_model1 will use MYRIAD, compiled_model2 will use CPU.
	 ************/

	// Example 2
	// Compile and load networks:
	ov::CompiledModel compiled_model3 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "LOW"}});
	ov::CompiledModel compiled_model4 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "MEDIUM"}});
	ov::CompiledModel compiled_model5 = core.compile_model(model, "AUTO:GPU,MYRIAD,CPU", {{ov::hint::model_priority.name(), "LOW"}});
	/************
	  Assume that all the devices (CPU, GPU, and MYRIAD) can support all the networks.
	  Result: compiled_model3 will use GPU, compiled_model4 will use GPU, compiled_model5 will use MYRIAD.
	 ************/
//! [part4]
	return 0;
}
