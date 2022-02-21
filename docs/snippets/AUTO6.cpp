#include <openvino/openvino.hpp>

int main() {
//! [part6]
   ov::Core core;

   // read a network in IR, PaddlePaddle, or ONNX format
   std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

   // load a network to AUTO and set log level to debug
   ov::CompiledModel compiled_model = core.compile_model(model, "AUTO", {{ov::log::level.name(), "LOG_DEBUG"}});

   // or set log level with set_config and load network
   core.set_property("AUTO", ov::log::level(ov::log::Level::DEBUG));
   ov::CompiledModel compiled_model2 = core.compile_model(model, "AUTO");

//! [part6]
    return 0;
}
