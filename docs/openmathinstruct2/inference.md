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
pip3 install tensorrt_llm==0.18.2 -U --pre --extra-index-url https://pypi.nvidia.com

# Check the installation was successful
python3 -c "import tensorrt_llm"
```

We also install Nemo-Skills.
```
pip install git+https://github.com/NVIDIA/NeMo-Skills.git@dh/fp8_v01
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
export HF_TOKEN=hf_Mt..

# Install the huggingface cli
pip install -U "huggingface_hub[cli]"

# Download the weights to a local directory
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir OpenMath-Nemotron-1.5B-Kaggle


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

LOCAL_DATASET_PATH = './calibrate_openmathreasoning'
prompt_template = get_prompt('generic/math', 'qwen-instruct')
calib_dataset = "nvidia/OpenMathReasoning"
calib_split = 'tir'
calib_size = 4096

# Load and take first N samples (no shuffling)
ds_samples = load_dataset(calib_dataset, split=calib_split, streaming=True)
ds_samples = list(islice(ds_samples, calib_size))

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

Now that our dataset is saved, let start calibration and conversion to `TensorRT-LLM` engine format.
We use a tensor parallelism setting of `--tp_size 2`, as we have two gpu case in our environment, feel free to

```
ns convert --input_model OpenMath-Nemotron-1.5B \
           --output_model OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm \
           --convert_from hf --convert_to trtllm \
           --num_gpus 2 \
           --dtype fp8 \
           --hf_model_name nvidia/OpenMath-Nemotron-1.5B \
           --model_type qwen \
           --calib_dataset calibrate_openmathreasoning
```
Now your engine is ready to be served.

#### 🏗️ ReDrafter speculative decoding (Work in Progress)

Planned work includes:
- Training a redrafter model
- Attaching weights to an FP8 quantized checkpoint


## 🏗️ Launching Servers (Work in Progress)

OpenMathReasoning is a tool-instruction reasoning model that combines an LLM with a code sandbox to answer questions. The system works as follows:

1. The LLM generates Python code wrapped in `<tool_call>` and `</tool_call>` tokens
2. The code is extracted and executed in the sandbox
3. The sandbox returns results or error traces
4. The output is fed back to the LLM for continued generation

An example below,
```bash
Therefore, b = k - 7 = 21 or 49. So, same result. Therefore, sum is 70.\n\nAlternatively, maybe I can write a small program to check for all bases b > 9, compute 9b + 7 and b + 7, check if the latter divides the former, and collect all such bases. Then sum them. Let\'s do that to verify.\n\nHere\'s a Python code to perform the check:\n\n<tool_call>\n# Initialize a list to store valid bases\nvalid_bases = []\n\n# Check bases from 10 upwards\nfor b in range(10, 10000):  # Arbitrary large upper limit\n    num1 = 9 * b + 7\n    num2 = b + 7\n    if num1 % num2 == 0:\n        valid_bases.append(b)\n        print(f"Found base: {b}")\n\n# Sum the valid bases\nsum_bases = sum(valid_bases)\nprint(f"Sum: {sum_bases}")\n\n# If sum is over 1000, take modulo 1000\nif sum_bases > 1000:\n    result = sum_bases % 1000\nelse:\n    result = sum_bases\n\nprint(f"Final Result: {result}")\n</tool_call>\n```output\nFound base: 21\nFound base: 49\nSum: 70\nFinal Result: 70\n```\nThe code confirms that the valid bases are 21 and 49, summing to 70.
```

### LLM server

!TODO - Install a fixed version of TRTLLM

As our instance has two servers we will use mpirun to launch the server and allocate 0.92 of the GPU memory.
```
mpirun --allow-run-as-root -n 2 python -m nemo_skills.inference.server.serve_trt \
    --model_path ./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm  --port 5000 --kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size
mpirun --allow-run-as-root -n 2 python -m nemo_skills.inference.server.serve_trt \
        --model_path ./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm  --host "127.0.0.1" --port 5000 --kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12
```
This will take a couple of minutes to start while your model loads, you should see logs finishing with the message, `INFO:     Uvicorn running on http://0.0.0.0:5000 ...`. The port and host can be seen in this message. Default local host is `0.0.0.0` and port specified above is `5000`.

If you run into problems and need to restart the server, you can kill existing `mpirun` processes with `pkill -9 -f mpirun`.

### Code execution server

To start the code sandbox simply run the below in a new terminal. It should run in non blocking mode.
```
python -m nemo_skills.code_execution.local_sandbox.local_sandbox_server &
```



```
# Backup commands - delete later
cd /mount/data/pkgs/aimo2/v01/
pip install flask ipython # push to main pending
mpirun --allow-run-as-root -n 2 ns start_server --model=./OpenMath-Nemotron-14B-Kaggle-fp8-trtllm  --server_gpus=2 \
    --server_type trtllm --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12" --with_sandbox
```

```
ns start_server --model=./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm --server_gpus=2 --server_type trtllm --with_sandbox \
    --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12 --host '127.0.0.1' --port 5000 "




ns start_server --model=./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm --server_gpus=2 --server_type trtllm \
    --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12 --host '127.0.0.1' --port 5000 "


ns start_server --model=./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm --server_gpus=2 --server_type trtllm \
    --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12 "


```

!TODO ... add in the step to point to this dataset.
https://nvidia.github.io/NeMo-Skills/openmathreasoning1/training/#download-data-and-convert-to-sft-format

### Async inference

Set and explain timeout time...

See kaggle notebook for eary stopping etc...


ns convert --input_model OpenMath-Nemotron-1.5B \
           --output_model OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm \
           --convert_from hf --convert_to trtllm \
           --num_gpus 2 \
           --dtype bf16 \
           --hf_model_name nvidia/OpenMath-Nemotron-1.5B \
           --model_type qwen