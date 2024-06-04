from torch.utils.data import Dataset
import json

## A custom JSON dataset loader, assumed that the dataset is a JSON blob. It should be an
## array of objects, the objects have a single conversation in the "text" property, e.g.:
##
## [
##   { "text": "<bos><start_of_turn>user Blah Blah Blah<end_of_sequence><start_of_turn>assistant Response Response<end_of_sequence>"}
## ]
##
## These conversations should be structured in the "chat template", which is basically:
##   * Starts with <bos> (beginning of sequence)
##   * Alternating between "user" and "assistant", a turn starts with "<start_of_turn>{ROLE}" and ends with "<end_of_turn>"
##
## I'd recommend that you split up conversations so they don't go too far over the "max_length". How you do that is up to you.
##
class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        text = sample["text"]
        # Tokenize the text using the provided tokenizer
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        # Return the tokenized text with key 'input_ids'
        return {"input_ids": inputs["input_ids"].flatten(), "attention_mask": inputs["attention_mask"].flatten()}
