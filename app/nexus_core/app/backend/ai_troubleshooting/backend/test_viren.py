# test_enhanced.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/viren_enhanced", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("models/viren_enhanced", local_files_only=True)

# Test the same questions
test_prompts = [
    "What are synthetic scenarios?",
    "How do you create training data?",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0])
    print(f"Q: {prompt}")
    print(f"A: {response}\n")