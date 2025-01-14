Image-to-Video synthesis with AnimateAnyone and OpenVINO
========================================================

|image0|

`AnimateAnyone <https://arxiv.org/pdf/2311.17117.pdf>`__ tackles the
task of generating animation sequences from a single character image. It
builds upon diffusion models pre-trained on vast character image
datasets.

The core of AnimateAnyone is a diffusion model pre-trained on a massive
dataset of character images. This model learns the underlying character
representation and distribution, allowing for realistic and diverse
character animation. To capture the specific details and characteristics
of the input character image, AnimateAnyone incorporates a ReferenceNet
module. This module acts like an attention mechanism, focusing on the
input image and guiding the animation process to stay consistent with
the original character’s appearance. AnimateAnyone enables control over
the character’s pose during animation. This might involve using
techniques like parametric pose embedding or direct pose vector input,
allowing for the creation of various character actions and movements. To
ensure smooth transitions and temporal coherence throughout the
animation sequence, AnimateAnyone incorporates temporal modeling
techniques. This may involve recurrent architectures like LSTMs or
transformers that capture the temporal dependencies between video
frames.

Overall, AnimateAnyone combines a powerful pre-trained diffusion model
with a character-specific attention mechanism (ReferenceNet), pose
guidance, and temporal modeling to achieve controllable, high-fidelity
character animation from a single image.

Learn more in `GitHub
repo <https://github.com/MooreThreads/Moore-AnimateAnyone>`__ and
`paper <https://arxiv.org/pdf/2311.17117.pdf>`__.

.. container:: alert alert-warning

   ::

      <p style="font-size:1.25em"><b>! WARNING !</b></p>
      <p>
          This tutorial requires at least <b>96 GB</b> of RAM for model conversion and <b>40 GB</b> for inference. Changing the values of <code>HEIGHT</code>, <code>WIDTH</code> and <code>VIDEO_LENGTH</code> variables will change the memory consumption but will also affect accuracy.
      </p>


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Prepare base model <#prepare-base-model>`__
-  `Prepare image encoder <#prepare-image-encoder>`__
-  `Download weights <#download-weights>`__
-  `Initialize models <#initialize-models>`__
-  `Load pretrained weights <#load-pretrained-weights>`__
-  `Convert model to OpenVINO IR <#convert-model-to-openvino-ir>`__

   -  `VAE <#vae>`__
   -  `Reference UNet <#reference-unet>`__
   -  `Denoising UNet <#denoising-unet>`__
   -  `Pose Guider <#pose-guider>`__
   -  `Image Encoder <#image-encoder>`__

-  `Inference <#inference>`__
-  `Video post-processing <#video-post-processing>`__
-  `Interactive inference <#interactive-inference>`__



This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/animate-anyone/animate-anyone.gif


Prerequisites
-------------



.. code:: ipython3

    from pathlib import Path
    import requests


    %pip install -q "torch>=2.1" torchvision einops omegaconf "diffusers<=0.24" "huggingface-hub<0.26.0" transformers av accelerate  "gradio>=4.19" --extra-index-url "https://download.pytorch.org/whl/cpu"
    %pip install -q "openvino>=2024.0" "nncf>=2.9.0"


    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)


    from cmd_helper import clone_repo

    clone_repo("https://github.com/itrushkin/Moore-AnimateAnyone.git")

    %load_ext skip_kernel_extension


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Note that we clone a fork of original repo with tweaked forward methods.

.. code:: ipython3

    MODEL_DIR = Path("models")
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
    REFERENCE_UNET_PATH = MODEL_DIR / "reference_unet.xml"
    DENOISING_UNET_PATH = MODEL_DIR / "denoising_unet.xml"
    POSE_GUIDER_PATH = MODEL_DIR / "pose_guider.xml"
    IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"

    WIDTH = 448
    HEIGHT = 512
    VIDEO_LENGTH = 24

    SHOULD_CONVERT = not all(
        p.exists()
        for p in [
            VAE_ENCODER_PATH,
            VAE_DECODER_PATH,
            REFERENCE_UNET_PATH,
            DENOISING_UNET_PATH,
            POSE_GUIDER_PATH,
            IMAGE_ENCODER_PATH,
        ]
    )

.. code:: ipython3

    from datetime import datetime
    from typing import Optional, Union, List, Callable
    import math

    from PIL import Image
    import openvino as ov
    from torchvision import transforms
    from einops import repeat
    from tqdm.auto import tqdm
    from einops import rearrange
    from omegaconf import OmegaConf
    from diffusers import DDIMScheduler
    from diffusers.image_processor import VaeImageProcessor
    from transformers import CLIPImageProcessor
    import torch

    from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
    from src.utils.util import get_fps, read_frames
    from src.utils.util import save_videos_grid
    from src.pipelines.context import get_context_scheduler


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/859/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/859/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(


.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    from pathlib import PurePosixPath
    import gc
    import warnings

    from typing import Dict, Any
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import CLIPVisionModelWithProjection
    import nncf

    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.models.pose_guider import PoseGuider


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino


Prepare base model
------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights/stable-diffusion-v1-5")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="botp/stable-diffusion-v1-5",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/3.44G [00:00<?, ?B/s]


Prepare image encoder
---------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    local_dir = Path("./pretrained_weights")
    local_dir.mkdir(parents=True, exist_ok=True)
    for hub_file in ["image_encoder/config.json", "image_encoder/pytorch_model.bin"]:
        saved_path = local_dir / hub_file
        if saved_path.exists():
            continue
        hf_hub_download(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            subfolder=PurePosixPath(saved_path.parent.name),
            filename=PurePosixPath(saved_path.name),
            local_dir=local_dir,
        )



.. parsed-literal::

    image_encoder/config.json:   0%|          | 0.00/703 [00:00<?, ?B/s]



.. parsed-literal::

    pytorch_model.bin:   0%|          | 0.00/1.22G [00:00<?, ?B/s]


Download weights
----------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse", local_dir="./pretrained_weights/sd-vae-ft-mse"
    )
    snapshot_download(
        repo_id="patrolli/AnimateAnyone",
        local_dir="./pretrained_weights",
    )



.. parsed-literal::

    Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/6.84k [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.46k [00:00<?, ?B/s]



.. parsed-literal::

    diffusion_pytorch_model.bin:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    Fetching 6 files:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/154 [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    motion_module.pth:   0%|          | 0.00/1.82G [00:00<?, ?B/s]



.. parsed-literal::

    reference_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    denoising_unet.pth:   0%|          | 0.00/3.44G [00:00<?, ?B/s]



.. parsed-literal::

    pose_guider.pth:   0%|          | 0.00/4.35M [00:00<?, ?B/s]


.. code:: ipython3

    config = OmegaConf.load("Moore-AnimateAnyone/configs/prompts/animation.yaml")
    infer_config = OmegaConf.load("Moore-AnimateAnyone/" + config.inference_config)

Initialize models
-----------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
    reference_unet = UNet2DConditionModel.from_pretrained(config.pretrained_base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256))
    image_enc = CLIPVisionModelWithProjection.from_pretrained(config.image_encoder_path)


    NUM_CHANNELS_LATENTS = denoising_unet.config.in_channels


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/859/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/modeling_utils.py:109: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      return torch.load(checkpoint_file, map_location="cpu")
    Some weights of the model checkpoint were not used when initializing UNet2DConditionModel:
     ['conv_norm_out.weight, conv_norm_out.bias, conv_out.weight, conv_out.bias']


Load pretrained weights
-----------------------



.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )


.. parsed-literal::

    <string>:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    <string>:6: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    <string>:9: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.


Convert model to OpenVINO IR
----------------------------

The pose sequence is initially
encoded using Pose Guider and fused with multi-frame noise, followed by
the Denoising UNet conducting the denoising process for video
generation. The computational block of the Denoising UNet consists of
Spatial-Attention, Cross-Attention, and Temporal-Attention, as
illustrated in the dashed box on the right. The integration of reference
image involves two aspects. Firstly, detailed features are extracted
through ReferenceNet and utilized for Spatial-Attention. Secondly,
semantic features are extracted through the CLIP image encoder for
Cross-Attention. Temporal-Attention operates in the temporal dimension.
Finally, the VAE decoder decodes the result into a video clip.

|image01|

The pipeline contains 6 PyTorch modules:

- VAE encoder
- VAE decoder
- Image encoder
- Reference UNet
- Denoising UNet
- Pose Guider

For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. models,
which require extensive memory to store the weights during inference,
can benefit from weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method. The main
difference between weights compression and full model quantization
(post-training quantization) is that activations remain floating-point
in the case of weights compression which leads to a better accuracy. In
addition, weight compression is data-free and does not require a
calibration dataset, making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters.

More details about weights compression can be found in `OpenVINO
documentation <https://docs.openvino.ai/2023.3/weight_compression.html>`__.

.. |image01| image:: https://humanaigc.github.io/animate-anyone/static/images/f2_img.png

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    warnings.simplefilter("ignore", torch.jit.TracerWarning)

VAE
~~~



The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder.

As the encoder and the decoder are used independently in different parts
of the pipeline, it will be better to convert them to separate models.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_ENCODER_PATH.exists():
        class VaeEncoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, x):
                return self.vae.encode(x).latent_dist.mean
        vae.eval()
        with torch.no_grad():
            vae_encoder = ov.convert_model(VaeEncoder(vae), example_input=torch.zeros(1,3,512,448))
        vae_encoder = nncf.compress_weights(vae_encoder)
        ov.save_model(vae_encoder, VAE_ENCODER_PATH)
        del vae_encoder
        cleanup_torchscript_cache()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (32 / 32)              │ 100% (32 / 32)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not VAE_DECODER_PATH.exists():
        class VaeDecoder(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, z):
                return self.vae.decode(z).sample
        vae.eval()
        with torch.no_grad():
            vae_decoder = ov.convert_model(VaeDecoder(vae), example_input=torch.zeros(1,4,HEIGHT//8,WIDTH//8))
        vae_decoder = nncf.compress_weights(vae_decoder)
        ov.save_model(vae_decoder, VAE_DECODER_PATH)
        del vae_decoder
        cleanup_torchscript_cache()
    del vae
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (40 / 40)              │ 100% (40 / 40)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Reference UNet
~~~~~~~~~~~~~~



Pipeline extracts reference attention features from all transformer
blocks inside Reference UNet model. We call the original forward pass to
obtain shapes of the outputs as they will be used in the next pipeline
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not REFERENCE_UNET_PATH.exists():
        class ReferenceUNetWrapper(torch.nn.Module):
            def __init__(self, reference_unet):
                super().__init__()
                self.reference_unet = reference_unet

            def forward(self, sample, timestep, encoder_hidden_states):
                return self.reference_unet(sample, timestep, encoder_hidden_states, return_dict=False)[1]

        sample = torch.zeros(2, 4, HEIGHT // 8, WIDTH // 8)
        timestep = torch.tensor(0)
        encoder_hidden_states = torch.zeros(2, 1, 768)
        reference_unet.eval()
        with torch.no_grad():
            wrapper =  ReferenceUNetWrapper(reference_unet)
            example_input = (sample, timestep, encoder_hidden_states)
            ref_features_shapes = {k: v.shape for k, v in wrapper(*example_input).items()}
            ov_reference_unet = ov.convert_model(
                wrapper,
                example_input=example_input,
            )
        ov_reference_unet = nncf.compress_weights(ov_reference_unet)
        ov.save_model(ov_reference_unet, REFERENCE_UNET_PATH)
        del ov_reference_unet
        del wrapper
        cleanup_torchscript_cache()
    del reference_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (270 / 270)            │ 100% (270 / 270)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Denoising UNet
~~~~~~~~~~~~~~



Denoising UNet is the main part of all diffusion pipelines. This model
consumes the majority of memory, so we need to reduce its size as much
as possible.

Here we make all shapes static meaning that the size of the video will
be constant.

Also, we use the ``ref_features`` input with the same tensor shapes as
output of `Reference UNet <#reference-unet>`__ model on the previous
step.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not DENOISING_UNET_PATH.exists():
        class DenoisingUNetWrapper(torch.nn.Module):
            def __init__(self, denoising_unet):
                super().__init__()
                self.denoising_unet = denoising_unet

            def forward(
                self,
                sample,
                timestep,
                encoder_hidden_states,
                pose_cond_fea,
                ref_features
            ):
                return self.denoising_unet(
                    sample,
                    timestep,
                    encoder_hidden_states,
                    ref_features,
                    pose_cond_fea=pose_cond_fea,
                    return_dict=False)

        example_input = {
            "sample": torch.zeros(2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "timestep": torch.tensor(999),
            "encoder_hidden_states": torch.zeros(2,1,768),
            "pose_cond_fea": torch.zeros(2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8),
            "ref_features": {k: torch.zeros(shape) for k, shape in ref_features_shapes.items()}
        }

        denoising_unet.eval()
        with torch.no_grad():
            ov_denoising_unet = ov.convert_model(
                DenoisingUNetWrapper(denoising_unet),
                example_input=tuple(example_input.values())
            )
        ov_denoising_unet.inputs[0].get_node().set_partial_shape(ov.PartialShape((2, 4, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        ov_denoising_unet.inputs[2].get_node().set_partial_shape(ov.PartialShape((2, 1, 768)))
        ov_denoising_unet.inputs[3].get_node().set_partial_shape(ov.PartialShape((2, 320, VIDEO_LENGTH, HEIGHT // 8, WIDTH // 8)))
        for ov_input, shape in zip(ov_denoising_unet.inputs[4:], ref_features_shapes.values()):
            ov_input.get_node().set_partial_shape(ov.PartialShape(shape))
            ov_input.get_node().set_element_type(ov.Type.f32)
        ov_denoising_unet.validate_nodes_and_infer_types()
        ov_denoising_unet = nncf.compress_weights(ov_denoising_unet)
        ov.save_model(ov_denoising_unet, DENOISING_UNET_PATH)
        del ov_denoising_unet
        cleanup_torchscript_cache()
    del denoising_unet
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (534 / 534)            │ 100% (534 / 534)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Pose Guider
~~~~~~~~~~~



To ensure pose controllability, a lightweight pose guider is devised to
efficiently integrate pose control signals into the denoising process.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not POSE_GUIDER_PATH.exists():
        pose_guider.eval()
        with torch.no_grad():
            ov_pose_guider = ov.convert_model(pose_guider, example_input=torch.zeros(1, 3, VIDEO_LENGTH, HEIGHT, WIDTH))
        ov_pose_guider = nncf.compress_weights(ov_pose_guider)
        ov.save_model(ov_pose_guider, POSE_GUIDER_PATH)
        del ov_pose_guider
        cleanup_torchscript_cache()
    del pose_guider
    gc.collect()


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (8 / 8)                │ 100% (8 / 8)                           │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Image Encoder
~~~~~~~~~~~~~



Pipeline uses CLIP image encoder to generate encoder hidden states
required for both reference and denoising UNets.

.. code:: ipython3

    %%skip not $SHOULD_CONVERT
    if not IMAGE_ENCODER_PATH.exists():
        image_enc.eval()
        with torch.no_grad():
            ov_image_encoder = ov.convert_model(image_enc, example_input=torch.zeros(1, 3, 224, 224), input=(1, 3, 224, 224))
        ov_image_encoder = nncf.compress_weights(ov_image_encoder)
        ov.save_model(ov_image_encoder, IMAGE_ENCODER_PATH)
        del ov_image_encoder
        cleanup_torchscript_cache()
    del image_enc
    gc.collect()


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/859/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:5006: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (146 / 146)            │ 100% (146 / 146)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Inference
---------



We inherit from the original pipeline modifying the calls to our models
to match OpenVINO format.

.. code:: ipython3

    core = ov.Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



For starting work, please select inference device from dropdown list.

.. code:: ipython3

    from notebook_utils import device_widget

    device = device_widget()

.. code:: ipython3

    class OVPose2VideoPipeline(Pose2VideoPipeline):
        def __init__(
            self,
            vae_encoder_path=VAE_ENCODER_PATH,
            vae_decoder_path=VAE_DECODER_PATH,
            image_encoder_path=IMAGE_ENCODER_PATH,
            reference_unet_path=REFERENCE_UNET_PATH,
            denoising_unet_path=DENOISING_UNET_PATH,
            pose_guider_path=POSE_GUIDER_PATH,
            device=device.value,
        ):
            self.vae_encoder = core.compile_model(vae_encoder_path, device)
            self.vae_decoder = core.compile_model(vae_decoder_path, device)
            self.image_encoder = core.compile_model(image_encoder_path, device)
            self.reference_unet = core.compile_model(reference_unet_path, device)
            self.denoising_unet = core.compile_model(denoising_unet_path, device)
            self.pose_guider = core.compile_model(pose_guider_path, device)
            self.scheduler = DDIMScheduler(**OmegaConf.to_container(infer_config.noise_scheduler_kwargs))

            self.vae_scale_factor = 8
            self.clip_image_processor = CLIPImageProcessor()
            self.ref_image_processor = VaeImageProcessor(do_convert_rgb=True)
            self.cond_image_processor = VaeImageProcessor(do_convert_rgb=True, do_normalize=False)

        def decode_latents(self, latents):
            video_length = latents.shape[2]
            latents = 1 / 0.18215 * latents
            latents = rearrange(latents, "b c f h w -> (b f) c h w")
            # video = self.vae.decode(latents).sample
            video = []
            for frame_idx in tqdm(range(latents.shape[0])):
                video.append(torch.from_numpy(self.vae_decoder(latents[frame_idx : frame_idx + 1])[0]))
            video = torch.cat(video)
            video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
            video = (video / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            video = video.cpu().float().numpy()
            return video

        def __call__(
            self,
            ref_image,
            pose_images,
            width,
            height,
            video_length,
            num_inference_steps=30,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "tensor",
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            context_schedule="uniform",
            context_frames=24,
            context_stride=1,
            context_overlap=4,
            context_batch_size=1,
            interpolation_factor=1,
            **kwargs,
        ):
            do_classifier_free_guidance = guidance_scale > 1.0

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            batch_size = 1

            # Prepare clip image embeds
            clip_image = self.clip_image_processor.preprocess(ref_image.resize((224, 224)), return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image)["image_embeds"]
            clip_image_embeds = torch.from_numpy(clip_image_embeds)
            encoder_hidden_states = clip_image_embeds.unsqueeze(1)
            uncond_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

            if do_classifier_free_guidance:
                encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                4,
                width,
                height,
                video_length,
                clip_image_embeds.dtype,
                torch.device("cpu"),
                generator,
            )

            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Prepare ref image latents
            ref_image_tensor = self.ref_image_processor.preprocess(ref_image, height=height, width=width)  # (bs, c, width, height)
            ref_image_latents = self.vae_encoder(ref_image_tensor)[0]
            ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)
            ref_image_latents = torch.from_numpy(ref_image_latents)

            # Prepare a list of pose condition images
            pose_cond_tensor_list = []
            for pose_image in pose_images:
                pose_cond_tensor = self.cond_image_processor.preprocess(pose_image, height=height, width=width)
                pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
                pose_cond_tensor_list.append(pose_cond_tensor)
            pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
            pose_fea = self.pose_guider(pose_cond_tensor)[0]
            pose_fea = torch.from_numpy(pose_fea)

            context_scheduler = get_context_scheduler(context_schedule)

            # denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                    counter = torch.zeros(
                        (1, 1, latents.shape[2], 1, 1),
                        device=latents.device,
                        dtype=latents.dtype,
                    )

                    # 1. Forward reference image
                    if i == 0:
                        ref_features = self.reference_unet(
                            (
                                ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                                torch.zeros_like(t),
                                # t,
                                encoder_hidden_states,
                            )
                        ).values()

                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            0,
                        )
                    )
                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                    context_queue = list(
                        context_scheduler(
                            0,
                            num_inference_steps,
                            latents.shape[2],
                            context_frames,
                            context_stride,
                            context_overlap,
                        )
                    )

                    num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                    global_context = []
                    for i in range(num_context_batches):
                        global_context.append(context_queue[i * context_batch_size : (i + 1) * context_batch_size])

                    for context in global_context:
                        # 3.1 expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        b, c, f, h, w = latent_model_input.shape
                        latent_pose_input = torch.cat([pose_fea[:, :, c] for c in context]).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)

                        pred = self.denoising_unet(
                            (
                                latent_model_input,
                                t,
                                encoder_hidden_states[:b],
                                latent_pose_input,
                                *ref_features,
                            )
                        )[0]

                        for j, c in enumerate(context):
                            noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                            counter[:, :, c] = counter[:, :, c] + 1

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            if interpolation_factor > 0:
                latents = self.interpolate_latents(latents, interpolation_factor, latents.device)
            # Post-processing
            images = self.decode_latents(latents)  # (b, c, f, h, w)

            # Convert to tensor
            if output_type == "tensor":
                images = torch.from_numpy(images)

            return images

.. code:: ipython3

    pipe = OVPose2VideoPipeline()

.. code:: ipython3

    pose_images = read_frames("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    src_fps = get_fps("Moore-AnimateAnyone/configs/inference/pose_videos/anyone-video-2_kps.mp4")
    ref_image = Image.open("Moore-AnimateAnyone/configs/inference/ref_images/anyone-5.png").convert("RGB")
    pose_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_list.append(pose_image_pil)

.. code:: ipython3

    video = pipe(
        ref_image,
        pose_list,
        width=WIDTH,
        height=HEIGHT,
        video_length=VIDEO_LENGTH,
    )



.. parsed-literal::

      0%|          | 0/30 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/24 [00:00<?, ?it/s]


Video post-processing
---------------------



.. code:: ipython3

    new_h, new_w = video.shape[-2:]
    pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
    pose_tensor_list = []
    for pose_image_pil in pose_images[:VIDEO_LENGTH]:
        pose_tensor_list.append(pose_transform(pose_image_pil))

    ref_image_tensor = pose_transform(ref_image)  # (c, h, w)
    ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
    ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
    pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
    pose_tensor = pose_tensor.transpose(0, 1)
    pose_tensor = pose_tensor.unsqueeze(0)
    video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

    save_dir = Path("./output")
    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    out_path = save_dir / f"{date_str}T{time_str}.mp4"
    save_videos_grid(
        video,
        str(out_path),
        n_rows=3,
        fps=src_fps,
    )

.. code:: ipython3

    from IPython.display import Video

    Video(out_path, embed=True)




.. raw:: html

    <video controls  >
     <source src="data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQABGJhtZGF0AAACuQYF//+13EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NCAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjQgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz04IGxvb2thaGVhZF90aHJlYWRzPTggc2xpY2VkX3RocmVhZHM9MSBzbGljZXM9OCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj0yNSBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNoPTAgcmNfbG9va2FoZWFkPTQwIHJjPWFiciBtYnRyZWU9MSBiaXRyYXRlPTEwMjQgcmF0ZXRvbD0xLjAgcWNvbXA9MC42MCBxcG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAAZ3ZYiEACD/2lu4PtiAGCZiIJmO35BneLS4/AKawbwF3gS81VgCN/Hryek5EZJp1IoIopMo/OyDntxcd3MAAAMAAAMAVxSBmCOAnDsVm8fhn7n0VHVyNk/lJFEHm3qFfS8vLgY7KTTJB/MAioDlK6EROKqL9bNritdbvzE2WXD2kjyA9/EeTazCG8xlnhyFUlgeczCK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" type="video/mp4">
     Your browser does not support the video tag.
     </video>



Interactive inference
---------------------



.. code:: ipython3

    import gradio as gr


    def generate(
        img,
        pose_vid,
        seed,
        guidance_scale,
        num_inference_steps,
        _=gr.Progress(track_tqdm=True),
    ):
        generator = torch.Generator().manual_seed(seed)
        pose_list = read_frames(pose_vid)[:VIDEO_LENGTH]
        video = pipe(
            img,
            pose_list,
            width=WIDTH,
            height=HEIGHT,
            video_length=VIDEO_LENGTH,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        new_h, new_w = video.shape[-2:]
        pose_transform = transforms.Compose([transforms.Resize((new_h, new_w)), transforms.ToTensor()])
        pose_tensor_list = []
        for pose_image_pil in pose_list:
            pose_tensor_list.append(pose_transform(pose_image_pil))

        ref_image_tensor = pose_transform(img)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=VIDEO_LENGTH)
        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)
        video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)

        save_dir = Path("./output/gradio")
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        out_path = save_dir / f"{date_str}T{time_str}.mp4"
        save_videos_grid(
            video,
            str(out_path),
            n_rows=3,
            fps=12,
        )
        return out_path

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/animate-anyone/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(fn=generate)

    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/"


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







