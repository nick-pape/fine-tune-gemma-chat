from InquirerPy import prompt
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from config import BASE_MODEL_NAME, NEW_MODEL_NAME, QUANTIZATION_CONFIG, SEQUENCE_LENGTH, GEMMA_CONFIG
from chat_history import ChatHistory

## Load up the base model
base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=QUANTIZATION_CONFIG,
        config=GEMMA_CONFIG
)

## Merge the base model with your trained model.
model = PeftModel.from_pretrained(base_model, './models/'+NEW_MODEL_NAME)

## Load up the base tokenizer, note we don't add EOS automatically here since we are generating.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, model_max_length=SEQUENCE_LENGTH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

history = ChatHistory(tokenizer)

while True:
    history.add_user_message(input("You: "))
    encoded = history.tokenize(tokenizer)

    raw_output = model.generate(
        **encoded,
        ## limit the response length, else it could just go on and on and on and on and on and on and.....
        max_new_tokens=100, 
        ## we only need one, but you can do more
        num_return_sequences=4, 
        do_sample=True,
        ## this is "beam search", which basically is a forking factor during generation. You can ramp this up, but it adds computation.
        num_beams=4, 
        ## this is sort of a hack, but basically I'm telling it that when we hit the first <end_of_turn> to treat it like an EOS.
        eos_token_id=107,
        ## tells the beams to stop searching if they hit EOS
        early_stopping=True, 
        ## you can read about these two... but basically they control the number of "possible sequences" during generation.
        top_k=100, 
        top_p=0.95,
        ## randomness factor
        temperature=0.7
    )

    ## this line will certainly make Pete tear his eyes out.
    ## but we are taking the first result (since num_return_sequences=1), and, since this includes the encoded,
    ## we basically lop the encoded off the front. We are relying on `eos_token_id` above to hope these are short.
    input_token_count = encoded['input_ids'].shape[1]
    responses = set([tokenizer.decode(i[input_token_count:], skip_special_tokens=True) for i in raw_output])

    if (len(responses) > 1):
        questions = [
            {
                "type": "list",
                "name": "selected_response",
                "message": "Select a response:",
                "choices": responses
            }
        ]
        answer = prompt(questions)
        selected = answer['selected_response']
    else:
        selected = responses.pop()

    history.add_bot_message(selected)

    os.system('cls' if os.name == 'nt' else 'clear')
    print(history)
