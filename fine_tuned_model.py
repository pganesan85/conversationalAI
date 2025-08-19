from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("distilbert-base-uncased")
print(model)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_answer_ft(query):
    inputs = tokenizer.encode(query, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
