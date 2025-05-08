# Inference engine

Here we demonstrate how to create a [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) inference engine. An inderence engine can be used to answer new unseen Math problems and can be hosted within a docker container or loacally. The example shall demonstrate a server in [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), however the same could be set up in any of the [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm) and [sglang](https://github.com/sgl-project/sglang) servers.

Add architecture pic...
![Alt text](../figs/inference_engine.png)

## TensorRT-LLM installation

For the inference, pretrained models were converted to TensorRT engines using TensorRT-LLM. TensorRT’s inflight batching boosts throughput by dynamically grouping inference requests, releasing each sample as soon as it completes—reducing latency and optimizing GPU utilization.

Detailed installation instuctions can be found [here](https://github.com/NVIDIA/TensorRT-LLM?tab=readme-ov-file#getting-started) and do note that if pytorch is in your environment it must be built with C++11 ABI on (see the installation [known limitations](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)).

For our example, we start with a `nvidia/cuda:12.8.1-devel-ubuntu24.04` container and install using the below (reference [here](https://github.com/nv-guomingz/TensorRT-LLM/blob/v0.14.0/docs/source/installation/linux.md#installing-on-linux)),
```
# Install prerequisites
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

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
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

from itertools import islice
from nemo_skills.prompt.utils import get_prompt
import os

N_CALIBRATION_SAMPLES = 4096
LOCAL_DATASET_PATH = './calibrate_openmathreasoning'
prompt_template = get_prompt('generic/math', 'qwen-instruct')

# Load and process dataset
all_splits = load_dataset("nvidia/OpenMathReasoning", streaming=True)
datasets = [all_splits[split] for split in all_splits.keys()]
ds = concatenate_datasets(datasets)
ds = ds.shuffle(seed=42, buffer_size=10000)
ds_samples = list(islice(ds, N_CALIBRATION_SAMPLES))

# Create a simple list of dictionaries with just the "text" field
texts = []
for sample in ds_samples:
    sample_dict = {k:v for k,v in sample.items() if k in ['problem', 'generation']}
    text = prompt_template.fill(sample_dict,
                        continue_prefix_generation=True,
                        prefix_generation_to_response=True)
    texts.append(text)  # Just store the text strings directly

# Create a minimal dataset with just the text column
calib_ds = Dataset.from_dict({"text": texts})  # Explicitly create a dict with "text" column

# Save in the correct format
os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)

# Save the dataset
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

Now we have a fp8 quantised checkpoint, so we can build our engine.
```
trtllm-build --checkpoint_dir $OpenMath-Nemotron-14B-Kaggle-fp8-ckpt \
    --output_dir $FP16_MODEL \
    --gemm_plugin  auto \
    --use_paged_context_fmha=enable \
    --max_batch_size 32 \
    --max_seq_len 24000 \
    --max_input_len 22000 \
    --max_num_tokens 22000 \
    --max_beam_width 1 \
    --kv_cache_type paged
```
Note, there is an open issue in using fp8 kv cache for this model so we do not use this.

```
trtllm-build --checkpoint_dir $TMP_FP8_MODEL \
    --output_dir $FP16_MODEL \
    --gemm_plugin  auto \
    --use_paged_context_fmha=enable \
    --max_batch_size 32 \
    --max_seq_len 49152 \
    --max_input_len 24576 \
    --max_num_tokens 1024 \
    --max_beam_width 1 \
    --kv_cache_type paged
```


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









