---
date: 2025-08-01
readtime: 20
---

# Building an Efficient Inference Engine for Math Problems

This tutorial guides you through creating a high-performance inference engine using [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) to tackle complex math problems. We'll leverage [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) for optimized model serving, including an advanced technique called ReDrafter for speculative decoding.

By the end of this tutorial, you'll have a local setup capable of running efficient inference with a large language model (LLM) integrated with a code execution sandbox. This setup is a simplified version of the pipeline that achieved success in the AIMO24 competition.

## What We'll Cover

1.  **Setting up Your Environment**: Get your system ready by installing necessary libraries within a suitable container.
2.  **Preparing Model Weights**: Download a pre-trained OpenMath model and convert it into an optimized TensorRT-LLM engine using FP8 quantization.
3.  **Accelerating Inference with ReDrafter**: Discover ReDrafter, a speculative decoding technique, train a draft model, and integrate it into our TensorRT-LLM engine for faster generation.
4.  **Launching the Inference Server**: Set up the LLM server and a parallel code execution sandbox to handle the tool-use capabilities of our model.
5.  **Running Inference**: Finally, we'll send math problems to our custom inference engine and observe its problem-solving abilities.

## TODOs
- change the Nemo-Skills installation (rn points to my fork)
- should we make the redrafter optional (not everyone might feel like training it)
- decide where to place the scripts for inference (stream generate)
- decide whether to keep the dataset creation script here or move it somewhere else

-----

## 1\. Setting Up Your Environment

Our first step is to establish a consistent and isolated environment. We'll use an NVIDIA PyTorch NGC container and install the essential libraries: TensorRT-LLM for model optimization and NeMo-Skills for the overall pipeline management.

### Container Setup and Library Installation

Once inside the `nvcr.io/nvidia/pytorch:25.05-py3` container, run the following commands to install TensorRT-LLM and NeMo-Skills:

```bash
# Ensure no conflicting TensorRT installations and install TensorRT-LLM
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt
pip3 install tensorrt_llm

# Install NeMo-Skills from the specified branch
pip install git+https://github.com/alessiodevoto/NeMo-Skills.git@aimo-inference
```

-----

## 2\. Preparing Model Weights

Now that our environment is ready, the next step is to prepare our Large Language Model (LLM). We'll download the `nvidia/OpenMath-Nemotron-14B-Kaggle` model and transform it into an optimized TensorRT-LLM engine using FP8 quantization. This process significantly improves inference speed and efficiency.

**Note on FP8 Quantization:** FP8 (8-bit floating point) quantization is highly efficient but requires GPUs that support `E4M3 FP8` (like NVIDIA Hopper GPUs). For other GPUs, `int8_wo` (8-bit integer with weight-only quantization) is recommended and doesn't require calibration.

### Downloading Model Weights and Dataset

Generate a Hugging Face token and export it as an environment variable, then use the Hugging Face CLI to download the necessary models and datasets.

```bash
# Export your Hugging Face token
export HF_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN # Replace with your actual token

# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Download the 14B parameter main model
huggingface-cli download nvidia/OpenMath-Nemotron-14B-kaggle --local-dir OpenMath-Nemotron-14B-kaggle

# Download a smaller model for ReDrafter training
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir OpenMath-Nemotron-1.5B

# Download the OpenMathReasoning dataset for calibration
huggingface-cli download nvidia/OpenMathReasoning --repo-type dataset --local-dir OpenMathReasoning
```

### Preparing the Calibration Dataset for FP8 Quantization

For FP8 quantization, a small calibration dataset is essential. We'll use a subset of the `OpenMathReasoning` dataset to create it. Save the following as `prepare_calibration_data.py`:

```python
import os
from datasets import load_dataset, Dataset
from itertools import islice
from nemo_skills.prompt.utils import get_prompt

# Define paths and parameters
LOCAL_DATASET_PATH = './calibration_dataset'
CALIB_DATASET_NAME = "nvidia/OpenMathReasoning"
CALIB_SPLIT = 'tir'
CALIB_SIZE = 4096

# Load samples, format them, and save as a Parquet file
print(f"Loading and formatting {CALIB_SIZE} samples for calibration...")
ds_samples = load_dataset(CALIB_DATASET_NAME, split=CALIB_SPLIT, streaming=True)
ds_samples = list(islice(ds_samples, CALIB_SIZE))

prompt_template = get_prompt('generic/math', 'qwen-instruct')
calibration_dataset = Dataset.from_dict({
    "text": [
        prompt_template.fill(
            {k: v for k, v in sample.items() if k in ['problem', 'generation']},
            continue_prefix_generation=True,
            prefix_generation_to_response=True
        )
        for sample in ds_samples
    ]
})

os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
calibration_dataset.to_parquet(f"{LOCAL_DATASET_PATH}/data.parquet")
print(f"Calibration dataset saved to {LOCAL_DATASET_PATH}/data.parquet")
```

Run the script:

```bash
python prepare_calibration_data.py
```

### Converting and Quantizing to TensorRT-LLM Engine

Now, convert the Hugging Face model to a TensorRT-LLM engine, applying FP8 quantization and using the prepared calibration dataset. This step generates the highly quantized LLM inference engine.

```bash
ns convert \
    --input_model OpenMath-Nemotron-14B-kaggle \
    --output_model OpenMath-Nemotron-14B-kaggle-fp8-trtllm \
    --convert_from hf \
    --convert_to trtllm \
    --num_gpus 1 \
    --dtype fp8 \
    --hf_model_name nvidia/OpenMath-Nemotron-14B-kaggle \
    --model_type qwen \
    --max_input_len 30000 \
    --max_seq_len 32000 \
    --no-trt_reuse_tmp_engine \
    --calib_dataset ./calibration_dataset
```

After this command, your main LLM engine is ready for deployment.

-----

## 3\. Accelerating Inference with ReDrafter

To push our inference efficiency further, we'll integrate [ReDrafter](https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm). This speculative decoding technique uses a smaller "draft" model to predict tokens, allowing the main LLM to generate responses much faster.

### Installing and Training ReDrafter

First, install the ReDrafter library. Then, we'll train the ReDrafter model using the `OpenMath-Nemotron-1.5B` model as its base and the `OpenMathReasoning` dataset.

```bash
# Install the ReDrafter library
pip install --no-binary=protobuf --ignore-requires-python \
        "git+https://github.com/apple/ml-recurrent-drafter.git#egg=recurrent-drafting[dev,train]"

# Train the ReDrafter model
ns run_cmd --log_dir ./logs/ \
torchrun --nproc_per_node=2 -m nemo_skills.training.train_redrafter \
    --llm_name_or_path 'OpenMath-Nemotron-1.5B' \
    --dataset "OpenMathReasoning" \
    --dataset_split "tir" \
    --bf16 True \
    --output_dir "redrafter_output" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --learning_rate 0.001 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 True \
    --model_max_length 2048 \
    --dataset_nrows 100000 \
    --drafter_predict_n_tokens 3 \
    --drafter_num_layers 2 \
    --rnn True \
    --phase train \
    --report_to wandb # Remove if not using wandb
```

During training, observe the `redrafter2_top1` score; aiming for above `0.6` indicates good performance (60% of steps accept the next three drafted tokens).

### Building the TensorRT-LLM Engine for the Draft Model

Now, we'll convert our trained ReDrafter model into a TensorRT-LLM checkpoint and then combine it with our main LLM to create the final, accelerated TensorRT-LLM engine.

First, clone the TensorRT-LLM repository to access its conversion scripts:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM/
```

Next, convert the trained ReDrafter PyTorch checkpoint to a TensorRT-LLM checkpoint.

```bash
export BASE_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-trtllm/tmp-ckpt
export REDRAFTER_PYTORCH_CKPT=$(pwd)/redrafter_output/redrafter_OpenMath-Nemotron-1.5B_n_3_lr_0.001_layers_2
export REDRAFTER_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-draft-ckpt

cd ./TensorRT-LLM/examples/redrafter
python convert_checkpoint.py \
    --base_model_checkpoint_dir $BASE_TRTLLM_CKPT \
    --drafter_model_dir $REDRAFTER_PYTORCH_CKPT \
    --output_dir $REDRAFTER_TRTLLM_CKPT \
    --dtype bfloat16 \
    --tp_size 1 \
    --redrafter_num_beams 1 \
    --redrafter_draft_len_per_beam 3
cd ../../../
```

Finally, build the combined TensorRT-LLM engine that includes both the main model and the ReDrafter for speculative decoding.

```bash
trtllm-build \
    --checkpoint_dir $REDRAFTER_TRTLLM_CKPT \
    --output_dir OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm \
    --gemm_plugin fp8 \
    --use_paged_context_fmha=enable \
    --max_batch_size 32 \
    --max_seq_len 32000 \
    --max_input_len 32000 \
    --max_num_tokens 32000 \
    --speculative_decoding_mode explicit_draft_tokens \
    --max_beam_width 1 \
    --kv_cache_type paged
```

Complete the setup by copying the tokenizer files:

```bash
cp OpenMath-Nemotron-14B-kaggle/*tok* OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm/
```

Your TensorRT-LLM engine, now supercharged with ReDrafter, is ready to be served\!

-----

## 4\. Launching the Inference Servers

Our LLM is a powerful tool-instruction reasoning model. This means it doesn't just generate text; it can also write and execute Python code in a secure sandbox to solve problems. This section details how to launch both the LLM server and its accompanying code execution sandbox.

The interaction works like this:

1.  The LLM generates Python code wrapped in `<tool_call>` and `</tool_call>` tokens.
2.  The inference engine extracts and sends this code to the sandbox.
3.  The sandbox executes the code and returns the results.
4.  The output is fed back to the LLM for continued generation or to finalize its answer.

Here's an example of such an interaction:

````bash
Therefore, b = k - 7 = 21 or 49. So, same result. Therefore, sum is 70.\n\nAlternatively, maybe I can write a small program to check for all bases b > 9, compute 9b + 7 and b + 7, check if the latter divides the former, and collect all such bases. Then sum them. Let\'s do that to verify.\n\nHere\'s a Python code to perform the check:\n\n<tool_call>\n# Initialize a list to store valid bases\nvalid_bases = []\n\n# Check bases from 10 upwards\nfor b in range(10, 10000):  # Arbitrary large upper limit\n    num1 = 9 * b + 7\n    num2 = b + 7\n    if num1 % num2 == 0:\n        valid_bases.append(b)\n        print(f"Found base: {b}")\n\n# Sum the valid bases\nsum_bases = sum(valid_bases)\nprint(f"Sum: {sum_bases}")\n\n# If sum is over 1000, take modulo 1000\nif sum_bases > 1000:\n    result = sum_bases % 1000\nelse:\n    result = sum_bases\n\nprint(f"Final Result: {result}")\n</tool_call>\n```output\nFound base: 21\nFound base: 49\nSum: 70\nFinal Result: 70\n```\nThe code confirms that the valid bases are 21 and 49, summing to 70.
````

### Starting the LLM and Sandbox Servers

Use the `ns start_server` command to launch both the LLM server and the code execution sandbox simultaneously.

```bash
mpirun -np 1 ns start_server \
    --model=./OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm/ \
    --server_gpus=1 \
    --server_type trtllm-serve \
    --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12" \
    --with_sandbox
```

Keep this terminal window open; the servers will run in the background.

-----

Here is a rewritten version of that section, with the code broken down and explained.

-----

## 5\. Running Inference

With our LLM and sandbox servers operational, we're ready for the final step: sending math problems to our powerful inference engine and observing its answers. NeMo-Skills provides convenient utility functions to interact with our hosted servers.

First, we'll initialize the necessary components to run our inference. This code sets up the sandbox environment and connects to the LLM server.

```python
import copy
from nemo_skills.inference.server.code_execution_model import CodeTags, get_code_execution_model
from nemo_skills.inference.server.server_utils import get_sandbox, stream_generate
from nemo_skills.prompt.utils import get_prompt
from datasets import load_dataset
import pandas as pd
from itertools import islice

# 1. Initialize the Sandbox and LLM Client
print("Initializing sandbox and LLM client...")
sandbox = get_sandbox()
llm = get_code_execution_model(server_type="trtllm-serve", sandbox=sandbox)
print("Sandbox and LLM client initialized.")
```

- We import several modules, including `get_sandbox` and `get_code_execution_model`, which are essential for setting up our environment.
- The `get_sandbox()` function creates a secure environment where the code from the LLM will be executed.
- `get_code_execution_model` creates a client that can communicate with our running LLM. We specify the `trtllm-serve` type to match the server we're using.


Next, we'll configure the prompt template and define the specific tags that the LLM will use to identify code blocks. This is how the LLM knows what part of its output should be sent to the sandbox.

```python
# 2. Prepare the Prompt Template and define code tags
print("Preparing prompt template and defining code tags...")
prompt_template = get_prompt('generic/math', 'qwen-instruct')
prompt_template.config.code_tags = CodeTags(
    code_begin="<tool_call>\n",
    code_end="</tool_call>\n",
)
```

These parameters control the generation process. Adjusting these values allows you to fine-tune the model's output, for instance, by controlling its creativity or verbosity.

```python
# 3. Define Sampling Parameters
sampling_params = {
    "tokens_to_generate": 8000,
    "temperature": 0.0,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    "max_code_executions": 2
}
print("Sampling parameters defined.")
```

Now we'll load some example math problems to test our setup. We're using a streaming dataset to efficiently load a small number of problems.

```python
# 4. Prepare Problems for Inference
problem = """
Three airline companies operate flights from Dodola island. Each company has a different schedule of departures. The first company departs every 100 days, the second every 120 days and the third every 150 days. 
What is the greatest positive integer $d$ for which it is true that there will be $d$ consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?
"""

request = copy.deepcopy(sampling_params)
request["prompts"] = [prompt_template.fill({'problem': problem})]
print(f"Prepared {len(list_of_problems)} problems for inference.")
```

This is the final step, where we send our prepared problems to the LLM and print the generated responses.

```python
# 5. Run Inference and process results
print("\n--- Starting Inference ---")
results = stream_generate(
    llm,
    **request,
    **prompt_template.get_code_execution_args(),
    stop_after_n_seconds=None,
    stop_after_n_completed=None,
    stop_after_n_same_answer=None
)

print(results)

print("\nInference complete. Your efficient math problem-solving engine is working!")
```

With `stream_generate` we send the problems to the LLM server and perform decoding in parallel.

Observe how the LLM generates responses, potentially including code blocks and their execution outputs, demonstrating its full problem-solving pipeline.