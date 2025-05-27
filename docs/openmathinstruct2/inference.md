# Inference Engine

This section demonstrates how to create a [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) inference engine for answering unseen math problems. The engine can be hosted locally or in a Docker container. While we show a [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) server example, similar setups work with [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm), or [sglang](https://github.com/sgl-project/sglang) servers.

![Inference Architecture](../figs/inference_engine.png)

## TensorRT-LLM Setup

We use TensorRT-LLM to convert pretrained models into TensorRT engines, leveraging inflight batching for improved throughput and latency.

### Installation

For TensorRT installation on linux, refer to the [installation guide](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).

In this example we work off the `nvcr.io/nvidia/pytorch:24.11-py3` container. Where we install by running the below,
```
# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt

<<<<<<< HEAD
# Install via pip and upgrade flash-attn
pip3 install tensorrt_llm==0.18.0 -U --pre --extra-index-url https://pypi.nvidia.com
pip3 install --upgrade flash-attn
=======
# Install tensorrt_llm
pip3 install tensorrt_llm==0.18.2 -U --pre --extra-index-url https://pypi.nvidia.com
>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032

# Check everyhting runs ok
python -c 'import tensorrt_llm'
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
<<<<<<< HEAD
huggingface-cli download nvidia/OpenMath-Nemotron-14B-kaggle --local-dir OpenMath-Nemotron-14B-kaggle
=======
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir OpenMath-Nemotron-1.5B-Kaggle


>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032
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

Now that our dataset is saved, let start calibration and conversion to `TensorRT-LLM` engine format. There are two steps run here, quantisation of the weights and saving them to a `TensorRT-LLM` pytorch checkpoint, and building the checkpoint into an C++ engine. You can also run the individual steps by following the `examples/models/` folder in `TensorRT-LLM` repo and looking at the options for each architecture.

<<<<<<< HEAD
Now lets start quantisation and calibration. The detailed steps are listed in TensorRT-LLM for different architectures, but here we will use a help command from NeMo-Skills.
Setting `--num_gpus 2` applies tensort pararallelism over the two gpu case in our environment. Teh

```
ns convert --input_model  OpenMath-Nemotron-14B-kaggle \
           --output_model OpenMath-Nemotron-14B-kaggle-fp8-trtllm \
           --convert_from hf \
           --convert_to trtllm \
           --num_gpus 2 \
           --dtype fp8 \
           --hf_model_name nvidia/OpenMath-Nemotron-14B-kaggle \
=======
We use a tensor parallelism setting of `--num_gpus 2`, as we have two gpu case in our environment - the model is split using tensor parallelism over these two gpus.

```
ns convert --input_model OpenMath-Nemotron-1.5B \
           --output_model OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm \
           --convert_from hf --convert_to trtllm \
           --num_gpus 2 \
           --dtype fp8 \
           --hf_model_name nvidia/OpenMath-Nemotron-1.5B \
>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032
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

<<<<<<< HEAD
NeMo-Skills provides a helper command with the ability to host as LLM server with the `--model` argument and a few extra arguments for the server config. A code execution server can also be hosted in parallel with the command `--with_sandbox`.
=======
!TODO - Install a fixed version of TRTLLM
>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032

As our instance has two servers we will use mpirun to launch the server and allocate 0.92 of the GPU memory.
```
<<<<<<< HEAD
ns start_server --model=./OpenMath-Nemotron-14B-kaggle-fp8-trtllm/
                --server_gpus=2 --server_type trtllm
                --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12"
                --with_sandbox
=======
mpirun --allow-run-as-root -n 2 python -m nemo_skills.inference.server.serve_trt \
    --model_path ./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm  --port 5000 --kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size
mpirun --allow-run-as-root -n 2 python -m nemo_skills.inference.server.serve_trt \
        --model_path ./OpenMath-Nemotron-1.5B-Kaggle-fp8-trtllm  --host "127.0.0.1" --port 5000 --kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12
>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032
```
This will take a couple of minutes to start while your model loads, you should see logs finishing with the message, `INFO:     Uvicorn running on http://0.0.0.0:5000 ...`. The port and host can be seen in this message. Default local host is `0.0.0.0` and port specified above is `5000`.

If you run into problems and need to restart the server, you can kill existing `mpirun` processes with `pkill -9 -f mpirun`.

### LLM generate

<<<<<<< HEAD

```
import copy
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.server.code_execution_model import get_code_execution_model
from nemo_skills.prompt.utils import get_prompt

sandbox = get_sandbox()  # localhost by default
llm = get_code_execution_model(server_type="trtllm", sandbox=sandbox)

prompt_template = get_prompt('generic/math', 'qwen-instruct')
prompt_template.config.template.code_begin = "<tool_call>\n"
prompt_template.config.template.code_end = "</tool_call>\n"
print(f'Prompt template : {prompt_template}')

print(repr(prompt_template.fill({'problem': 'What is 1+1?'})))

sampling_params = {
    "tokens_to_generate": 8000,
    "temperature": 0.,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
}

request = copy.deepcopy(sampling_params)
list_of_texts = [prompt_template.fill({'problem': 'What is 1+1?'})]*4
request["prompts"] = list_of_texts

output = llm.generate(**request, **prompt_template.get_code_execution_args())

output[0]['generation']


prompt = prompt_template.fill({'problem': 'What is the sum of all prime numbers less than 10 million?'})
request["prompts"] = [prompt] * 4

output = llm.generate(**request, **prompt_template.get_code_execution_args())

output[0]['generation']



```
=======
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
>>>>>>> fca0fd6676cbdf73b326a94c6fb73540c39fa032

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