import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def generate_completion(model, tokenizer, prompt: str, device, max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


def few_shot_arithmetic(model, tokenizer, device):
    examples = """
Q: 2 + 3 = ?
A: 5

Q: 10 + 15 = ?
A: 25

Q: 7 + 8 = ?
A: 15

Q: 123 + 456 = ?
A: 579

"""
    query = "Q: 34 + 89 = ?\nA:"
    prompt = examples + query
    completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens=16)
    print("Arithmetic prompt:\n", prompt)
    print("Model answer:", completion)


def few_shot_sentiment(model, tokenizer, device):
    examples = """
Review: "The movie was fantastic, I loved every minute."
Sentiment: positive

Review: "It was boring and a complete waste of time."
Sentiment: negative

Review: "The acting was decent but the plot was weak."
Sentiment: mixed

"""
    query = 'Review: "I would definitely watch it again, it was great."\nSentiment:'
    prompt = examples + query
    completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens=8)
    print("\nSentiment prompt:\n", prompt)
    print("Model answer:", completion)


def few_shot_translation(model, tokenizer, device):
    examples = """
English: "Hello, how are you?"
French: "Bonjour, comment ça va ?"

English: "I love machine learning."
French: "J'aime l'apprentissage automatique."

English: "This book is very interesting."
French: "Ce livre est très intéressant."

"""
    query = 'English: "The weather is nice today."\nFrench:'
    prompt = examples + query
    completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens=32)
    print("\nTranslation prompt:\n", prompt)
    print("Model answer:", completion)


def few_shot_qa(model, tokenizer, device):
    examples = """
Q: What is the capital of France?
A: Paris

Q: Who wrote the play 'Romeo and Juliet'?
A: William Shakespeare

Q: What is the largest planet in our solar system?
A: Jupiter

"""
    query = "Q: What is the chemical symbol for water?\nA:"
    prompt = examples + query
    completion = generate_completion(model, tokenizer, prompt, device, max_new_tokens=8)
    print("\nQA prompt:\n", prompt)
    print("Model answer:", completion)


def main():
    device = configure_device()
    model_name = "gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    few_shot_arithmetic(model, tokenizer, device)
    few_shot_sentiment(model, tokenizer, device)
    few_shot_translation(model, tokenizer, device)
    few_shot_qa(model, tokenizer, device)


if __name__ == "__main__":
    main()
