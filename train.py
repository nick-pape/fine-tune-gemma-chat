from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
from chat_dataset import CustomDataset
from config import BASE_MODEL_NAME, SEQUENCE_LENGTH, DATA_FILE_PATH, NEW_MODEL_NAME, QUANTIZATION_CONFIG, ACCESS_TOKEN, GEMMA_CONFIG

## Load up the tokenizer (which converts words/word parts into indices in a dictionary)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, token=ACCESS_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_eos_token = True

## Load up your JSON blob
dataset = CustomDataset(DATA_FILE_PATH, tokenizer, SEQUENCE_LENGTH)

## Load up the model using HuggingFace magic libraries.
## Automatically sent to GPU.
## (btw check "torch.cuda.is_available()" to make sure GPU is working)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    config=GEMMA_CONFIG,
    quantization_config=QUANTIZATION_CONFIG,
    low_cpu_mem_usage=True,
    device_map="auto",
    token=ACCESS_TOKEN
)

## Some random settings for training. I don't really know what they do.
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

## This is a weird thing called "LoRA" or "Low Rank Adaptation".
## Basically, what we are doing here is not training the full model.
## We are only training certain layers (target_modules).
## We also set dropout (which means randomly deactivating some neurons during training,
## it's common practice to improve model stability).
## I wouldn't mess with these until reading up more on them.
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj']
)
model = get_peft_model(model, peft_config)

## Arguments for the training utility.
## The important ones here are num_train_epochs (the number of times it goes through your training set).
## And learning_rate (initial learning rate). There's no science here. Picking these is an art.
## These values worked decently on my training set of ~350 conversations.
training_arguments = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=1,
    output_dir="./models/"+NEW_MODEL_NAME,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    fp16=False,
    bf16=False,
    group_by_length=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False} # this fixes another warning
)

## The actual training utility.
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=SEQUENCE_LENGTH,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

## Do the training!! This could take some time :). Model saved to disk afterwards.
trainer.train()
trainer.model.save_pretrained('./models/' + NEW_MODEL_NAME)