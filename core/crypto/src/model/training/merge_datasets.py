import json
from pathlib import Path

def merge_all_to_single_jsonl():
    """Merge all JSONL files into one big file"""
    datasets_dir = Path("datasets")
    output_file = Path("datasets/merged_dataset.jsonl")
    
    all_lines = []
    
    # Find all JSONL files
    jsonl_files = list(datasets_dir.glob("**/*.jsonl"))
    print(f"🦍 Found {len(jsonl_files)} JSONL files")
    
    for jsonl_file in jsonl_files:
        print(f"📖 Reading: {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_lines.extend(lines)
            print(f"   Added {len(lines)} lines")
    
    # Write merged file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(all_lines)
    
    print(f"✅ Merged {len(all_lines)} total lines into: {output_file}")
    return output_file

if __name__ == "__main__":
    merge_all_to_single_jsonl()