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

On command line, generate a huggingface key to download the weights. Make sure the key has write access so you can host the huggingface calibration dataset.
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
from datasets import load_dataset, concatenate_datasets, Dataset
from itertools import islice
from nemo_skills.prompt.utils import get_prompt

N_CALIBRATION_SAMPLES = 4096
NEW_DATASET = 'darragh/calibrate_openmathreasoning'
prompt_template = get_prompt('generic/math', 'qwen-instruct')

# Load all splits (default splits are 'cot', 'tir', 'genselect')
all_splits = load_dataset("nvidia/OpenMathReasoning", streaming=True)
datasets = [all_splits[split] for split in all_splits.keys()]  # Get all splits

# Concatenate into a single IterableDataset
ds = concatenate_datasets(datasets)

# Shuffle and take first N samples
ds = ds.shuffle(seed=42, buffer_size=10000)
ds_samples = list(islice(ds, N_CALIBRATION_SAMPLES))

# Add the problems and answer into the prompt template
texts = []
for sample in ds_samples:
    sample_dict = {k:v for k,v in sample.items() if k in ['problem', 'generation']}
    text = prompt_template.fill(sample_dict,
                        continue_prefix_generation=True,
                        prefix_generation_to_response=True)
    texts.append({"text": text})

# Push to the hub
calib_ds = Dataset.from_list(texts)
calib_ds.push_to_hub(NEW_DATASET)
```

The above calibration dataset is created at

Now lets start calibration. !!IN WORK!!
```
python quantize.py --model_dir $BASE_MODEL \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --output_dir $TMP_FP8_MODEL \
                                   --calib_size 4636 \
                                   --calib_dataset $CALIB_DATASET \
                                   --batch_size 4 \
                                   --tp_size 8
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









