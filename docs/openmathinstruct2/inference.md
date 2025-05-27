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

# Install via pip and upgrade flash-attn
pip3 install tensorrt_llm==0.18.0 -U --pre --extra-index-url https://pypi.nvidia.com
pip3 install --upgrade flash-attn

# Check everyhting runs ok
python -c 'import tensorrt_llm'
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
huggingface-cli download nvidia/OpenMath-Nemotron-14B-kaggle --local-dir OpenMath-Nemotron-14B-kaggle
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
           --model_type qwen \
           --calib_dataset calibrate_openmathreasoning
```
Now your engine is ready to be served.

#### 🏗️ ReDrafter speculative decoding (Work in Progress)

Planned work includes:
- Training a redrafter model
- Attaching weights to an FP8 quantized checkpoint


## 🏗️ Launch servers (Work in Progress)

NeMo-Skills provides a helper command with the ability to host as LLM server with the `--model` argument and a few extra arguments for the server config. A code execution server can also be hosted in parallel with the command `--with_sandbox`.

```
ns start_server --model=./OpenMath-Nemotron-14B-kaggle-fp8-trtllm/
                --server_gpus=2 --server_type trtllm
                --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12"
                --with_sandbox
```

If you run into problems and need to restart the server, you can kill existing `mpirun` processes with `pkill -9 -f mpirun`.

### LLM generate


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

### Async inference

Set and explain timeout time...

See kaggle notebook for eary stopping etc...









