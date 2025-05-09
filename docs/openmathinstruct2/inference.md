# Inference Engine

This section demonstrates how to create a [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) inference engine for answering unseen math problems. The engine can be hosted locally or in a Docker container. While we show a [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) server example, similar setups work with [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm), or [sglang](https://github.com/sgl-project/sglang) servers.

![Inference Architecture](../figs/inference_engine.png)

## TensorRT-LLM Setup

We use TensorRT-LLM to convert pretrained models into TensorRT engines, leveraging inflight batching for improved throughput and latency.

### Installation

1. Start from `nvidia/cuda:12.8.1-devel-ubuntu24.04` container
2. Follow the [installation guide](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#getting-started)

Note: If using PyTorch, ensure it's built with C++11 ABI enabled (see [known limitations](https://github.com/nv-guomingz/TensorRT-LLM/blob/v0.14.0/docs/source/installation/linux.md#installing-on-linux)),
```
# Install prerequisites
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs vim

# Install tensorrt_llm
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check the installation was successful
python3 -c "import tensorrt_llm"
```

We also install Nemo-Skills.
```
pip install git+https://github.com/NVIDIA/NeMo-Skills.git
python3 -c "import nemo_skills"
```

## Prepare weights

You can find the `OpenMathReasoning` collection [here](https://huggingface.co/collections/nvidia/openmathreasoning-68072c0154a5099573d2e730). We will use the `nvidia/OpenMath-Nemotron-14B-Kaggle` model with 14B parameters.

For this demonstration we will use FP8 quantisation, you can see the TensorRT-LLM Quantization Toolkit Installation Guide for more information. Note, FP8 quantisation is limited to GPUs which support `E4M3 FP8`, such as the Hopper family. For other GPUs, we recommend using `int8_wo` as mentioned in the Quantization Toolkit Guide. Calibration is not required for `int8_wo`.

!TODO: Mention not to import tensorrt in the main script because .... (check why this was)

### Download weights and dataset

On command line, generate a huggingface key to download the weights.
```
# Export your huggingface key to an environment variable
export HF_TOKEN=hf_Mt...

# Install the huggingface cli
pip install -U "huggingface_hub[cli]"

# Download the weights to a local directory
huggingface-cli download nvidia/OpenMath-Nemotron-14B-Kaggle --local-dir OpenMath-Nemotron-14B-Kaggle
```

For calibration of the weights during quantisation we also need a dataset. We will use the dataset the models were trained on.
```
huggingface-cli download nvidia/OpenMathReasoning --repo-type dataset
```

#### FP8 Quantization

In a python script or Jupyter notebook, generate the calibrarion dataset.
```
import os
from datasets import load_dataset, Dataset
from itertools import islice
from nemo_skills.prompt.utils import get_prompt

N_CALIBRATION_SAMPLES = 4096
LOCAL_DATASET_PATH = './calibrate_openmathreasoning'
prompt_template = get_prompt('generic/math', 'qwen-instruct')

# Load and take first N samples (no shuffling)
ds_samples = list(islice(
    load_dataset("nvidia/OpenMathReasoning", split='tir', streaming=True),
    N_CALIBRATION_SAMPLES
))

# Create dataset with formatted text
calib_ds = Dataset.from_dict({
    "text": [
        prompt_template.fill(
            {k: v for k, v in sample.items() if k in ['problem', 'generation']},
            continue_prefix_generation=True,
            prefix_generation_to_response=True
        )
        for sample in ds_samples
    ]
})

# Save
os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calib_ds.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")
```

Now that our dataset is saved, let start calibration.

Now lets start quantisation and calibration.
We clone `TensorRT-LLM` to use the helper scripts in model preparation.
We use a tensor parallelism setting of `--tp_size 2`, as we have two gpu case in our environment, feel free to

```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM/examples/quantization/

python quantize.py --model_dir ../../../OpenMath-Nemotron-14B-Kaggle \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --output_dir ../../../OpenMath-Nemotron-14B-Kaggle-fp8-ckpt \
                                   --calib_size 4096 \
                                   --calib_dataset ../../../calibrate_openmathreasoning \
                                   --batch_size 4 \
                                   --tp_size 2
cd ../../../
```

Now we have a fp8 quantised checkpoint, so we can build our engine. Note, there is an open issue in using fp8 kv cache for this model so we do not use this. If you do not use ReDrafter, set `--max_num_tokens 1000`. This is related to the chunked context which is incompatible currently with ReDrafter.
```
# Build the engine
trtllm-build --checkpoint_dir OpenMath-Nemotron-14B-Kaggle-fp8-ckpt \
    --output_dir OpenMath-Nemotron-14B-Kaggle-fp8-trtllm \
    --gemm_plugin  auto \
    --use_paged_context_fmha=enable \
    --max_batch_size 32 \
    --max_seq_len 24000 \
    --max_input_len 22000 \
    --max_num_tokens 22000 \
    --max_beam_width 1 \
    --kv_cache_type paged

# Copy the tokenizers to the engine directory
cp OpenMath-Nemotron-14B-Kaggle/*tok* OpenMath-Nemotron-14B-Kaggle-fp8-trtllm/
```

Now your engine is ready to be served.

#### 🏗️ ReDrafter speculative decoding (Work in Progress)

Planned work includes:
- Training a redrafter model
- Attaching weights to an FP8 quantized checkpoint


## 🏗️ Launch servers (Work in Progress)

!TODO - PR Flask import
!TODO - Install a fixed version of TRTLLM

```
cd /mount/data/pkgs/aimo2/v01/
mpirun --allow-run-as-root -n 2 ns start_server --model=./OpenMath-Nemotron-14B-Kaggle-fp8-trtllm  --server_gpus=2 \
    --server_type trtllm --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12" --with_sandbox

mpirun --allow-run-as-root -n 2 python -m nemo_skills.inference.server.serve_trt \
    --model_path ./OpenMath-Nemotron-14B-Kaggle-fp8-trtllm  --port 5000 --kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12
```

If you run into problems and need to restart the server, you can kill existing `mpirun` processes with `pkill -9 -f mpirun`.

### Code execution server

...

### LLM server

...

### Async inference

Set and explain timeout time...

See kaggle notebook for eary stopping etc...









