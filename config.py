## After requesting access to the Gemma model here: https://huggingface.co/google/gemma-7b-it
## Get your Hugging Face access token and put it below: https://huggingface.co/settings/tokens
## (Alternately, remove this variable and usage below and use HF_TOKEN environment variable)
access_token = "..."

## The name of the base model. For gemma chat scenarios, stick with "gemma-7b-it" or "gemma-2b-it".
BASE_MODEL_NAME = "google/gemma-7b-it"
## The name of the model you are creating. A folder will be created with the model after training.
NEW_MODEL_NAME = "pete-bot"
## The path to your training data (see "chat_dataset.py")
DATA_FILE_PATH = "data.json"
## The maximum sequence length to do training on (in tokens). I've been using 512 and 1024, but you can go larger.
SEQUENCE_LENGTH = 1024


## This is optional. But basically, we are attempting to "quantize" the model.
## Which basically means we reduce the weights from 16 (or 32) bit floats.
## This reduces the model's footprint in memory, and in some cases, can make inference faster.
## Some models come with 32 bits FP, but it's kind of unnecessary, 16 bit FP does well.
## 8-bit integers also does well.
## 4-bit integers also do surprisingly well.
## This config attempts to load in 4-bit integer (although I think it falls back to FP16).
## You may need to remove this if you don't have a GPU with CUDA-12 (it can't do 4-bit).
import torch
from transformers import BitsAndBytesConfig
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)