# Fine-tuning Gemma

## Setup
1. Install python 3.11
    * Note: don't use 3.12, otherwise you will get a warning about FlashAttention not being compiled. It's stupid,
    but I set it up this way because I know how you feel about warnings.
1. Ensure you have CUDA drivers
    * You'll need the right version for your GPU. And you may need to update `requirements.txt` with the torch cuda version.
        * 12.1 is the latest I've been using (RTX 3090)
    * NVIDIA download link: https://developer.nvidia.com/cuda-toolkit-archive
    * Run `nvidia-smi` to verify.
1. Run `pip install -r requirements.txt`
1. Run the following to ensure Python is able to find CUDA:
    ```python
    import torch
    torch.cuda.is_available()
    ```
1. Get access to Gemma and generate a Hugging Face access token. (See `config.py`).

## Files
* `config.py` - various configuration options
* `train.py` - trains a model
* `chat.py` - runs a chat loop on a fine-tuned model
* `chat_dataset.pt` - helper for loading a JSON blob of messages

## Usage
1. Create a dataset. Update `config.py` to point at your dataset.
1. Run `python train.py`
1. Run `python chat.py` to do an interactive chat loop.