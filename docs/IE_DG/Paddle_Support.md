# PaddlePaddle Support in OpenVINO™ {#openvino_docs_IE_DG_Paddle_Support}

Starting from the 2022.1 release, OpenVINO™ supports reading native PaddlePaddle models.
The `Core::ReadNetwork()` method provides a uniform way to read models from either the PaddlePaddle format or IR, which is the recommended approach.

## Read PaddlePaddle Models from IR

A PaddlePaddle Model can be read after it is [converted](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md) to [Intermediate Representation (IR)](../MO_DG/IR_and_opsets.md).

@sphinxdirective
.. tab:: C++ example

   .. code-block:: cpp
   
      InferenceEngine::Core core;
      auto network = core.ReadNetwork("model.xml");

.. tab:: Python example

   .. code-block:: python
   
      from openvino.inference_engine import IECore
      ie = IECore()
      net = ie.read_network("model.xml")
@endsphinxdirective

## Read PaddlePaddle Models from The Paddle Format (PaddlePaddle "inference model" model type)

@sphinxdirective
.. tab:: C++ example

   .. code-block:: cpp

      InferenceEngine::Core core;
      auto network = core.ReadNetwork("model.pdmodel");

.. tab:: Python example

   .. code-block:: cpp

      from openvino.inference_engine import IECore
      ie = IECore()
      net = ie.read_network("model.pdmodel")
@endsphinxdirective

**The Reshape feature:**

OpenVINO™ does not provide a mechanism to specify pre-processing, such as mean values subtraction or reverse input channels, for the PaddlePaddle format.
If a PaddlePaddle model contains dynamic shapes for input, use the `CNNNetwork::reshape` method for shape specialization.


> **NOTE**:
> PaddlePaddle [`inference model`](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/inference_en.md) mainly contains two kinds of files: `model.pdmodel`(model file) and `model.pdiparams`(params file), which are used for inference.
>
> The list of supported PaddlePaddle models and a description of how to export them can be found in [Convert a PaddlePaddle Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle.md). The following PaddlePaddle models are supported by intel CPU only: `Fast-SCNN`, `Yolo v3`, `ppyolo`, `MobileNetv3-SSD`, `BERT`.
>
> To `Normalize` PaddlePaddle Models, input data should be in FP32 format.
> When reading PaddlePaddle models from the PaddlePaddle format, make sure that `model.pdmodel` and `model.pdiparams` are in the same folder directory.
