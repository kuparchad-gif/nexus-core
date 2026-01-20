# mega_merger.py
import torch
from transformers import AutoModelForCausalLM

def mega_merge():
    print("🦍 MEGA-MERGING ALL WEIGHTS INTO VIREN")
    
    # Load Viren
    model = AutoModelForCausalLM.from_pretrained("models/viren_compactifai")
    
    # Load all extracted weights
    all_weights = torch.load("extracted_weights.pth")
    
    merged_count = 0
    for source_name, weights in all_weights.items():
        print(f"🔄 Merging from: {source_name}")
        
        # Different merging strategies for different sources
        if 'h5' in source_name:
            # Keras weights
            merged_count += merge_keras_weights(model, weights)
        elif 'ipynb' in source_name:
            # Code knowledge
            merged_count += merge_code_knowledge(model, weights)
        elif 'jsonl' in source_name:
            # Training data patterns
            merged_count += merge_training_patterns(model, weights)
    
    print(f"🎯 MEGA-MERGED {merged_count} KNOWLEDGE SOURCES")
    model.save_pretrained("models/viren_mega")
    
    return "models/viren_mega"

# Run the mega merger
mega_path = mega_merge()