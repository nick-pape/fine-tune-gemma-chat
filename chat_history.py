from config import SEQUENCE_LENGTH

## Chat history helper class.
class ChatHistory:
    def __init__(self, tokenizer, max_tokens = SEQUENCE_LENGTH - 100):
        self.__messages = []
        self.__tokenizer = tokenizer
        self.max_tokens = max_tokens

    def add_user_message(self, message):
        self.__messages.append({"role": "user", "content": message})

        while (len(self.tokenize(self.__tokenizer)['input_ids'][0]) > self.max_tokens):
            self.__messages.pop(0)
            self.__messages.pop(0)
    
    def add_bot_message(self, message):
        self.__messages.append({"role": "assistant", "content": message})

    def to_chat_format(self):
        return self.__tokenizer.apply_chat_template(self.__messages, tokenize=False, add_generation_prompt=True)

    def tokenize(self, tokenizer):
        prompt = self.to_chat_format()
        return tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=SEQUENCE_LENGTH).to("cuda")
    
    def __str__(self):
        return '\n'.join([
            ('You: ' if msg['role'] == 'user' else 'Model: ') + msg['content']
            for msg in self.__messages
        ])