from config import SEQUENCE_LENGTH

## Chat history helper class.
class ChatHistory:
    def __init__(self, tokenizer):
        self.__messages = []
        self.__tokenizer = tokenizer

    def add_user_message(self, message):
        self.__messages.append({"role": "user", "content": message})
    
    def add_bot_message(self, message):
        self.__messages.append({"role": "assistant", "content": message})

    def to_chat_format(self):
        return self.__tokenizer.apply_chat_template(self.__messages, tokenize=False, add_generation_prompt=True)

    def tokenize(self, tokenizer):
        prompt = self.to_chat_format()
        return tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=SEQUENCE_LENGTH).to("cuda")