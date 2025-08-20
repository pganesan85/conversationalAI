from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("distilbert-base-uncased")
print(model)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_answer_ft(query, model, tokenizer):
    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Encode safely
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)

    # Case 1: Generative models (Causal LM / Seq2Seq LM)
    if isinstance(model, (AutoModelForCausalLM, AutoModelForSeq2SeqLM)):
        outputs = model.generate(**inputs, max_length=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    # Case 2: Classification model
    elif isinstance(model, AutoModelForSequenceClassification):
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
        return f"Predicted class: {predicted_class_id}"

    # Fallback
    else:
        return "Unsupported model type for fine-tuned answering."
