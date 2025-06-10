# Inference Engine

This section demonstrates how to create a [NeMo-Skills](https://nvidia.github.io/NeMo-Skills/) inference engine for answering unseen math problems. The engine can be hosted locally or in a Docker container. While we show a [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) server example, similar setups work with [NeMo](https://github.com/NVIDIA/NeMo), [vLLM](https://github.com/vllm-project/vllm), or [sglang](https://github.com/sgl-project/sglang) servers.

![Inference Architecture](../figs/inference.png)

## TensorRT-LLM Setup

We use TensorRT-LLM to convert pretrained models into TensorRT engines, leveraging inflight batching for improved throughput and latency.

### Installation 🏗️ while waiting for PR merge (build from my PR repo)

For TensorRT installation on linux, refer to the [installation guide](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).

In this example we work off the `nvcr.io/nvidia/pytorch:25.01-py3` container. Where we install by running the below,
```
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/darraghdog/TensorRT-LLM.git -b dh/aimo2_v01
cd TensorRT-LLM


# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt
TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0rc0-cp312-cp312-linux_x86_64.whl pip install -e .

# Check everything runs ok
python -c 'import tensorrt_llm'
```

### Installation

For TensorRT installation on linux, refer to the [installation guide](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).

In this example we work off the `nvcr.io/nvidia/pytorch:25.01-py3` container. Where we install by running the below,
```
# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt

pip3 install tensorrt_llm==0.21.0rc0 -U --pre --extra-index-url https://pypi.nvidia.com

# Check everything runs ok
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
export HF_TOKEN=hf_Mt..

# Install the huggingface cli
pip install -U "huggingface_hub[cli]"

# Download the weights to a local directory
huggingface-cli download nvidia/OpenMath-Nemotron-14B-kaggle --local-dir OpenMath-Nemotron-14B-kaggle
huggingface-cli download nvidia/OpenMath-Nemotron-1.5B --local-dir OpenMath-Nemotron-1.5B
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

## ReDrafter speculative decoding

[ReDrafter](https://machinelearning.apple.com/research/redrafter-nvidia-tensorrt-llm) uses an RNN draft model, and combines beam search with dynamic tree attention to speed up LLM token generation by up to 3.5 tokens per generation step for open source models, surpassing the performance of prior speculative decoding techniques.

### ReDrafter Training

To train ReDrafter for the `OpenMath-Nemotron-1.5B` model we run the below. We train below on the same dataset the model was trained on. If this data is not available the redrafter could alternatively be trained on prompts and generations form your model. Feel free to test different parameters, however we found 20k samples or more was sufficent to train a good ReDrafter.
```
# Install the ReDrafter library, we are on a later version of python so can ignore that check.
pip install --no-binary=protobuf --ignore-requires-python \
        "git+https://github.com/apple/ml-recurrent-drafter.git#egg=recurrent-drafting[dev,train]"

# Train
ns run_cmd --log_dir ./logs/ \
torchrun --nproc_per_node=2 -m nemo_skills.training.train_redrafter \
    --llm_name_or_path 'OpenMath-Nemotron-1.5B' \
    --bf16 True \
    --output_dir "redrafter_" \
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
    --report_to wandb
```

You can add W&B logging by setting `--report_to wandb`. Intermittent logs should be reported to screen like below, ideally you will be reaching a `redrafter2_top1` score over `0.6`, which means in 60% of the steps the next three tokens are accepted from the draft model.

```
{'redrafter0_top1': 0.8328, 'redrafter0_top2': 0.9262, 'redrafter0_top3': 0.9545, 'redrafter0_top4': 0.9731, 'redrafter0_top5': 0.9785, 'redrafter0_loss': 0.5236, 'redrafter1_top1': 0.7697, 'redrafter1_top2': 0.8895, 'redrafter1_top3': 0.9301, 'redrafter1_top4': 0.9477, 'redrafter1_top5': 0.9589, 'redrafter1_loss': 0.7651, 'redrafter2_top1': 0.7123, 'redrafter2_top2': 0.8425, 'redrafter2_top3': 0.8953, 'redrafter2_top4': 0.93, 'redrafter2_top5': 0.9481, 'redrafter2_loss': 0.936, 'epoch': 1.0}
```

Below is an example of the acceptance rate during training. Note, as we only use one epoch we do not evaluate on a separate validation set.

![ReDrafter training](../figs/redrafter_training.png)

### Building TensorRT-LLM engine for draft model

Clone `TensorRT-LLM` so we have the examples with conversion scripts.
```
git clone https://github.com/darraghdog/TensorRT-LLM/ -b dh/aimo2_v01
```
When `NeMo-Skills` quantises a model we get a pytorch quantised checkpoint which is used to build the engine. We shall use this checkpoint to build a engine with `ReDrafter`. From the quantisation above, your checkpoint should be named something like, `OpenMath-Nemotron-14B-kaggle-fp8-trtllm-tmp-ckpt`.

We create the ReDrafter model using our trained model. Ensure that the `dtype` of your drafter is the same as the converted base model.
```
export BASE_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-trtllm-tmp-ckpt
export REDRAFTER_PYTORCH_CKPT=$(pwd)/redrafter__redrafter_OpenMath-Nemotron-14B-kaggle_n_3_lr_0.001_layers_2
export REDRAFTER_TRTLLM_CKPT=$(pwd)/OpenMath-Nemotron-14B-kaggle-fp8-draft-ckpt
cd ./TensorRT-LLM/examples/redrafter
python convert_checkpoint.py --base_model_checkpoint_dir $BASE_TRTLLM_CKPT \
                             --drafter_model_dir $REDRAFTER_PYTORCH_CKPT \
                             --output_dir $REDRAFTER_TRTLLM_CKPT \
                             --dtype bfloat16 \
                             --tp_size 2 \
                             --redrafter_num_beams 1 \
                             --redrafter_draft_len_per_beam 3
cd ../../../
```

Now we build the checkpoint. Here we need to use the `trtllm-build` command. As we pass token sequences form the model to the sandbox and back, even if the initial math question is short, we need a long input length allowed to accomodate the llm generation plus the executed code.

```
trtllm-build --checkpoint_dir $REDRAFTER_TRTLLM_CKPT \
    --output_dir OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm \
    --gemm_plugin fp8 \
     --use_paged_context_fmha=enable \
     --max_batch_size 32 \
     --max_seq_len 32000 \
     --max_input_len  32000 \
     --max_num_tokens 32000 \
     --speculative_decoding_mode explicit_draft_tokens \
     --max_beam_width 1 \
     --kv_cache_type paged
```

And finally we copy the tokenizer.
```
# Copy the tokenizer
cp OpenMath-Nemotron-14B-kaggle/*tok* OpenMath-Nemotron-14B-kaggle-fp8-redrafter-trtllm/
```


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

NeMo-Skills provides a helper command with the ability to host as LLM server with the `--model` argument and a few extra arguments for the server config. A code execution server can also be hosted in parallel with the command `--with_sandbox`.

As our instance has two servers we will use mpirun to launch the server and allocate 0.92 of the GPU memory.
```
ns start_server --model=./OpenMath-Nemotron-14B-kaggle-fp8-trtllm/
                --server_gpus=2 --server_type trtllm
                --server_args "--kv_cache_free_gpu_memory_fraction=0.92 --max_batch_size 12"
                --with_sandbox
```

### 🏗️ LLM generate



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

In this example we work off the `nvcr.io/nvidia/pytorch:25.01-py3` container.
```
# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt

pip3 install tensorrt_llm==0.21.0rc0 -U --pre --extra-index-url https://pypi.nvidia.com

# Check everything runs ok
python -c 'import tensorrt_llm'
```


### MISC - 🏗️ To be deleted later

```
mkdir /mount/data/pkgs/aimo2/v11
cd /mount/data/pkgs/aimo2/v11


export BASEDIR=/mount/data/pkgs/aimo2/v11/
export VICDIR=${BASEDIR}/vicuna-7b-v1.3/
export VICREDIR=${BASEDIR}/vicuna-7b-v1.3-redrafter/

git clone https://github.com/NVIDIA/TensorRT-LLM.git

cd TensorRT-LLM/examples/redrafter/

python convert_checkpoint.py --model_dir $VICDIR \
                             --drafter_model_dir $VICREDIR \
                             --output_dir ./tllm_checkkpoint_1gpu_redrafter \
                             --dtype bfloat16 \
                             --redrafter_num_beams 4 \
                             --redrafter_draft_len_per_beam 5


trtllm-build --checkpoint_dir ./tllm_checkkpoint_1gpu_redrafter \
             --output_dir ./tmp/redrafter/7B/trt_engines/fp16/1-gpu/ \
             --gemm_plugin bfloat16 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_batch_size 4

```




```

echo $HISTSIZE $HISTFILESIZE
export HISTSIZE=50000
export HISTFILESIZE=100000


mkdir /mount/data/pkgs/aimo2/v17
cd /mount/data/pkgs/aimo2/v17
ls -lahtr ./
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/darraghdog/TensorRT-LLM.git -b dh/aimo2_v01
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt
TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0rc0-cp312-cp312-linux_x86_64.whl pip install -e .



# Check everything runs ok
python -c 'import tensorrt_llm'


cp -r /mount/data/pkgs/aimo2/v06/vicuna-7b-v1.3/ .
cp -r /mount/data/pkgs/aimo2/v06/2025-06-04_13:32:16_redrafter_vicuna-7b-v1.3_n_5_lr_0.001_layers_2/ .
cp -r /mount/data/pkgs/aimo2/v06/Qwen2.5-7B-Instruct/ .
cp -r /mount/data/pkgs/aimo2/v06/redrafter__redrafter_Qwen2.5-7B-Instruct_n_3_lr_0.001_layers_2/ .

cd $BASEDIR
ln -s /mount/data/pkgs/aimo2/v06/vicuna-7b-v1.3
ln -s /mount/data/pkgs/aimo2/v06/2025-06-04_13:32:16_redrafter_vicuna-7b-v1.3_n_5_lr_0.001_layers_2/ vicuna-7b-v1.3-redrafter

export BASEDIR=/mount/data/pkgs/aimo2/v17/
export VICDIR=${BASEDIR}/vicuna-7b-v1.3/
export VICCKPTDIR=${BASEDIR}/vicuna-7b-v1.3-ckpt/
export VICREDIR=${BASEDIR}/vicuna-7b-v1.3-redrafter/
export VICRECKPTDIR=${BASEDIR}/vicuna-7b-v1.3-redrafter-ckpt/
export VICRETRTDIR=${BASEDIR}/vicuna-7b-v1.3-redrafter-trtllm/

cd ${BASEDIR}/TensorRT-LLM/examples/models/core/llama/
python convert_checkpoint.py --model_dir $VICDIR \
                              --output_dir $VICCKPTDIR \
                              --dtype float16

cd ${BASEDIR}/TensorRT-LLM/examples/redrafter/
python convert_checkpoint.py --base_model_checkpoint_dir $VICCKPTDIR \
                             --drafter_model_dir $VICREDIR \
                             --output_dir $VICRECKPTDIR \
                             --dtype float16 \
                             --redrafter_num_beams 6 \
                             --redrafter_draft_len_per_beam 6

cd ${BASEDIR}/
trtllm-build --checkpoint_dir $VICRECKPTDIR \
             --output_dir $VICRETRTDIR \
             --gemm_plugin float16 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_beam_width 1 \
             --max_batch_size 4

cd ${BASEDIR}/TensorRT-LLM/examples/redrafter/
python ../run.py --engine_dir $VICRETRTDIR \
                 --tokenizer_dir $VICDIR \
                 --max_output_len=100 \
                 --input_text "Once upon" "The basic idea of a Transformer model is"



[06/09/2025-12:36:12] [TRT] [W] ReDrafterForLLaMALM/_fwd_helper_L75/_process_logits_and_hidden_states_L737/_process_gen_logits_L600/_validate_draft_tokens_L87/elementwise_binary_L3011/ELEMENTWISE_LESS_0: dimensions not compatible for elementwise.
[06/09/2025-12:36:12] [TRT] [W] Was not able to infer a kOPT value for tensor ReDrafterForLLaMALM/_fwd_helper_L75/_process_logits_and_hidden_states_L693/sum_L3274/reduce_L3193/REDUCE_SUM_0_output_0. Using one(s).
[06/09/2025-12:36:12] [TRT] [W] Was not able to infer a kOPT value for tensor ReDrafterForLLaMALM/_fwd_helper_L75/_process_logits_and_hidden_states_L721/_get_gen_token_indices_for_unpack_L620/max_L3249/reduce_L3193/REDUCE_MAX_0_output_0. Using one(s).
[06/09/2025-12:36:12] [TRT] [E] IBuilder::buildSerializedNetwork: Error Code 4: Internal Error (kOPT values for profile 0 violate shape constraints: ReDrafterForLLaMALM/_fwd_helper_L75/_process_logits_and_hidden_states_L737/_process_gen_logits_L600/_validate_draft_tokens_L80/slice_L1323/SLICE_0: ISliceLayer has out of bounds access on axis 0 Out of bounds access for slice.)
Traceback (most recent call last):
  File "/usr/local/bin/trtllm-build", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/mount/data/pkgs/aimo2/v14/TensorRT-LLM/tensorrt_llm/commands/build.py", line 626, in main
    parallel_build(model_config, ckpt_dir, build_config, args.output_dir,
  File "/mount/data/pkgs/aimo2/v14/TensorRT-LLM/tensorrt_llm/commands/build.py", line 419, in parallel_build
    passed = build_and_save(rank, rank % workers, ckpt_dir,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mount/data/pkgs/aimo2/v14/TensorRT-LLM/tensorrt_llm/commands/build.py", line 384, in build_and_save
    engine = build_model(build_config,
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mount/data/pkgs/aimo2/v14/TensorRT-LLM/tensorrt_llm/commands/build.py", line 377, in build_model




















cd $BASEDIR
ln -s /mount/data/pkgs/aimo2/v06/Qwen2.5-7B-Instruct
ln -s /mount/data/pkgs/aimo2/v06/redrafter__redrafter_Qwen2.5-7B-Instruct_n_3_lr_0.001_layers_2/ Qwen2.5-7B-Instruct-redrafter





QWDIR=${BASEDIR}/Qwen2.5-7B-Instruct/
QWCKPTDIR=${BASEDIR}/Qwen2.5-7B-Instruct-ckpt/
QWREDIR=${BASEDIR}/Qwen2.5-7B-Instruct-redrafter/
QWRECKPTDIR=${BASEDIR}/Qwen2.5-7B-Instruct-redrafter-ckpt/
QWRETRTDIR=${BASEDIR}/Qwen2.5-7B-Instruct-redrafter-trtllm/

cd ${BASEDIR}/TensorRT-LLM/examples/models/core/qwen/
python ../../../quantization/quantize.py --model_dir $QWDIR \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --output_dir $QWCKPTDIR \
                                   --calib_size 512


cd ${BASEDIR}/TensorRT-LLM/examples/redrafter/
python convert_checkpoint.py --base_model_checkpoint_dir $QWCKPTDIR \
                             --drafter_model_dir $QWREDIR \
                             --output_dir $QWRECKPTDIR \
                             --dtype float16 \
                             --redrafter_num_beams 6 \
                             --redrafter_draft_len_per_beam 6

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir $QWRECKPTDIR \
             --output_dir $QWRETRTDIR \
             --gemm_plugin fp8 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_beam_width 6 \
             --max_batch_size 4


python ../run.py --engine_dir $QWRETRTDIR \
                 --tokenizer_dir $QWDIR \
                 --max_output_len=100 \
                 --input_text "Once upon" "The basic idea of a Transformer model is"









export BASEDIR=/mount/data/pkgs/aimo2/v10/
export VICDIR=${BASEDIR}/vicuna-7b-v1.3/
export VICCKPTDIR=${BASEDIR}/vicuna-7b-v1.3-ckpt/
export VICREDIR=${BASEDIR}/2025-06-04_13:32:16_redrafter_vicuna-7b-v1.3_n_5_lr_0.001_layers_2/
export VICRECKPTDIR_66=${BASEDIR}/vicuna-7b-v1.3-redrafter-66/
export VICRETRTDIR_66=${BASEDIR}/vicuna-7b-v1.3-redrafter-trtllm-66/


cd ${BASEDIR}/TensorRT-LLM/examples/redrafter/

# From the `examples/redrafter/` directory, run,
python convert_checkpoint.py --base_model_checkpoint_dir $VICCKPTDIR \
                             --drafter_model_dir $VICREDIR \
                             --output_dir $VICRECKPTDIR_66 \
                             --dtype float16 \
                             --redrafter_num_beams 6 \
                             --redrafter_draft_len_per_beam 6

cd ${BASEDIR}/
trtllm-build --checkpoint_dir $VICRECKPTDIR_66 \
             --output_dir $VICRETRTDIR_66 \
             --gemm_plugin float16 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_beam_width 6 \
             --max_batch_size 4

python run.py --engine_dir $VICRETRTDIR_66 \
                 --tokenizer_dir $VICDIR \
                 --max_output_len=100 \
                 --input_text "Once upon" "The basic idea of a Transformer model is"



QWDIR=${BASEDIR}/Qwen2.5-7B-Instruct/
QWCKPTDIR=${BASEDIR}/Qwen2.5-7B-Instruct-ckpt/
QWREDIR=${BASEDIR}/redrafter__redrafter_Qwen2.5-7B-Instruct_n_3_lr_0.001_layers_2/
QWRECKPTDIR_66=${BASEDIR}/Qwen2.5-7B-Instruct-redrafter/
QWRETRTDIR_66=${BASEDIR}/Qwen2.5-7B-Instruct-redrafter-trtllm/


cd ${BASEDIR}/TensorRT-LLM/examples/redrafter/
# From the `examples/redrafter/` directory, run,
python convert_checkpoint.py --base_model_checkpoint_dir $QWCKPTDIR \
                             --drafter_model_dir $QWREDIR \
                             --output_dir $QWRECKPTDIR_66 \
                             --dtype float16 \
                             --redrafter_num_beams 6 \
                             --redrafter_draft_len_per_beam 6

# Build trtllm engines from the trtllm checkpoint
# Enable fp8 context fmha to get further acceleration by setting `--use_fp8_context_fmha enable`
trtllm-build --checkpoint_dir $QWRECKPTDIR_66 \
             --output_dir $QWRETRTDIR_66 \
             --gemm_plugin fp8 \
             --speculative_decoding_mode explicit_draft_tokens \
             --max_beam_width 1 \
             --max_batch_size 4


python run.py --engine_dir $QWRETRTDIR \
                 --tokenizer_dir $QWDIR \
                 --max_output_len=100 \
                 --input_text "Once upon" "The basic idea of a Transformer model is"




```


```
cd /mount/data/pkgs/aimo2/v08
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/darraghdog/TensorRT-LLM.git -b dh/aimo2_v01
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

# See section `Install inside the PyTorch NGC Container`
[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt
pip uninstall -y tensorrt

# https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-2-python-only-build-without-c-compilation
TRTLLM_PRECOMPILED_LOCATION=https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-0.21.0rc0-cp312-cp312-linux_x86_64.whl pip install -e .
TRTLLM_USE_PRECOMPILED=1 pip wheel . --no-deps --wheel-dir ./build --pre --extra-index-url https://pypi.nvidia.com



math-dh-ngc-2501

```



```
# To kill existing mpirun
pkill -9 -f mpirun


huggingface-cli download lmsys/vicuna-7b-v1.3 --local-dir vicuna-7b-v1.3

huggingface-cli download Qwen/Qwen-7B-Chat --local-dir Qwen-7B-Chat
huggingface-cli download  Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct
cd /mount/data/pkgs/aimo2/v06/
git clone https://github.com/apple/ml-recurrent-drafter.git
cd ml-recurrent-drafter/
pip install --no-binary=protobuf --ignore-requires-python -e .
cd /mount/data/pkgs/aimo2/v06/ml-recurrent-drafter/recurrent_drafting/cmd

cd ../../

ln -s /mount/data/pkgs/aimo2/v06/Qwen-7B/
ln -s /mount/data/pkgs/aimo2/v06/vicuna-7b-v1.3/
ln -s /mount/data/pkgs/aimo2/v06/Qwen-7B-Chat/

# change n_proc to 2;
    change epochs to 0.8
    and model to vicuna-7b-v1.3;
    remove `set -e`;
    remove the ['--evaluation_strategy', 'no'] line
    set num_proc=16 at the bottom of train.py
vim recurrent_drafting/cmd/train.sh
wandb login --relogin
pip install transformers==4.45.2 sentence-transformers==3.1.1
./recurrent_drafting/cmd/train.sh

# For qwen only, in train.py do the below when loading the tokenizer and add `trust_remote_code=True`
tokenizer.pad_token = '<|extra_0|>'
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')


# At the end
pip install --upgrade  transformers==4.51.1


vim /usr/local/lib/python3.12/dist-packages/nemo_skills/training/train_redrafter.py
ns run_cmd --log_dir ./logs/ \
torchrun --nproc_per_node=2 -m nemo_skills.training.train_redrafter \
    --llm_name_or_path 'Qwen2.5-7B-Instruct' \
    --dataset_name tttonyyy/DeepScale-qwen2.5_7b-multi_16k \
    --dataset_split 'train' \
    --bf16 True \
    --output_dir "redrafter_" \
    --num_train_epochs 6 \
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
    --dataset_nrows 400000 \
    --drafter_predict_n_tokens 3 \
    --drafter_num_layers 2 \
    --rnn True \
    --phase train \
    --report_to wandb
```