# viren_model_test.py
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model = AutoModelForCausalLM.from_pretrained(
        "./models/viren_compactifai", 
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "./models/viren_compactifai",
        local_files_only=True
    )
    
    print("✅ VIREN LOADED SUCCESSFULLY!")
    
    # Test generation
    input_text = "Hello Viren"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0])
    
    print(f"Viren says: {response}")
    
except Exception as e:
    print(f"❌ Still failing: {e}")