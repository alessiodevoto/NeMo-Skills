# Inference engine

Here we demonstrate how to create a [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) inference engine. An inderence engine can be used to answer new unseen Math problems and can be hosted within a docker container or loacally. The example shall demonstrate a server in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), however the same could be set up in any of the [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm) and [sglang](https://github.com/sgl-project/sglang) servers.

Add architecture pic...
![Alt text](../figs/inference_engine.png)


## TensorRT-LLM installation

... example installation within a pytorch ngc container; talking with trtllm team ...
... Install nemo-skill ...

## Prepare weights

...

### Download weights

...
```
pip install -U "huggingface_hub[cli]"
huggingface-cli download "nvidia/OpenMath-Nemotron-14B-Kaggle" --local-dir OpenMath-Nemotron-14B-Kaggle
```

### Accelerate weights

A simple example could be set up with BF16, but ... talk about AIMO acceleration, etc.

#### INT8 Quantization

...

#### FP8 Quantization

...

#### Optionally add speculative decoding

Create a calibration dataset & as well as (have we released any open math generations).
Train a redrafter model.
Attach weights to an FP8 quantized checkpoint.


## Launch servers

...

### Code execution server

...

### LLM server

...

### Async inference

Set and explain timeout time...

See kaggle notebook for eary stopping etc...









