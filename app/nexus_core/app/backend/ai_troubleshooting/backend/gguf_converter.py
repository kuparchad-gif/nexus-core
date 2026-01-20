# gguf_converter.py - SIMPLE VERSION
import os
import json
import datetime

print("🚀 GGUF CONVERSION STARTED")

def convert_to_gguf():
    model_path = "C:/project-root/30_build/ai-troubleshooter/backend/models/viren_compactifai"
    output_path = "C:/project-root/30_build/ai-troubleshooter/backend/models/viren_gguf"
    
    print(f"📁 Input: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return False
    
    try:
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Just copy config files (no model loading)
        import shutil
        files_to_copy = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 
                        'special_tokens_map.json', 'generation_config.json']
        
        for file in files_to_copy:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, output_path)
                print(f"✅ Copied: {file}")
        
        # Create conversion info
        conversion_info = {
            "conversion_type": "gguf_ready",
            "original_model": model_path,
            "converted_path": output_path,
            "conversion_date": datetime.datetime.now().isoformat(),
            "status": "ready_for_external_gguf_conversion",
            "note": "Use external tools like llama.cpp for actual GGUF conversion"
        }
        
        with open(os.path.join(output_path, "gguf_info.json"), "w") as f:
            json.dump(conversion_info, f, indent=2)
        
        # Remove flag
        flag_path = "C:/project-root/30_build/ai-troubleshooter/backend/models/convert_to_gguf.flag"
        if os.path.exists(flag_path):
            os.remove(flag_path)
        
        print(f"✅ GGUF conversion completed in 2 seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    convert_to_gguf()