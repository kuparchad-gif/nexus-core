# brutal_merge.py
import os
import json
from pathlib import Path

def brutal_file_reader(file_path):
    """READ ANY FILE - DON'T CARE ABOUT FORMAT"""
    try:
        # Method 1: Try reading as text with multiple encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                if content.strip():
                    return content
            except:
                continue
        
        # Method 2: Read as binary and decode what we can
        with open(file_path, 'rb') as f:
            binary_content = f.read()
            # Try to decode as much as possible
            text_content = binary_content.decode('utf-8', errors='ignore')
            if text_content.strip():
                return text_content
        
        # Method 3: If all else fails, treat as JSONL-like and extract lines
        lines = []
        with open(file_path, 'rb') as f:
            for line in f:
                try:
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    if decoded_line:
                        lines.append(decoded_line)
                except:
                    continue
        return '\n'.join(lines)
        
    except Exception as e:
        return f"// ERROR reading {file_path}: {str(e)}"

def extract_jsonl_content(content, file_path):
    """EXTRACT ANY JSON-LIKE CONTENT"""
    lines = content.split('\n')
    jsonl_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to parse as JSON
        try:
            json.loads(line)
            jsonl_lines.append(line)
            continue
        except:
            pass
            
        # If it looks like JSON object but malformed, try to fix
        if '{' in line and '}' in line:
            try:
                # Extract between first { and last }
                start = line.find('{')
                end = line.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = line[start:end]
                    json.loads(json_str)
                    jsonl_lines.append(json_str)
                    continue
            except:
                pass
                
        # If it's not JSON, treat as text content and create JSON object
        if len(line) > 10:  # Only include substantial lines
            json_obj = {
                "text": line,
                "source_file": str(file_path),
                "type": "raw_text"
            }
            jsonl_lines.append(json.dumps(json_obj))
    
    return jsonl_lines

def brutal_merge_all_data():
    """MERGE EVERY FILE IN DATASETS FOLDER"""
    datasets_dir = Path("datasets")
    output_file = Path("datasets/brutal_merged.jsonl")
    
    all_jsonl_lines = []
    
    # Find ALL files recursively
    all_files = list(datasets_dir.glob("**/*"))
    print(f"🦍 FOUND {len(all_files)} TOTAL FILES")
    
    for file_path in all_files:
        if file_path.is_dir():
            continue
            
        print(f"🔨 SMASHING: {file_path.name}")
        
        try:
            # Read whatever the fuck is in there
            content = brutal_file_reader(file_path)
            
            # Extract JSONL content
            jsonl_lines = extract_jsonl_content(content, file_path)
            
            if jsonl_lines:
                all_jsonl_lines.extend(jsonl_lines)
                print(f"   ✅ EXTRACTED {len(jsonl_lines)} LINES")
            else:
                print(f"   ⚠️  NO CONTENT EXTRACTED")
                
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            continue
    
    # Write merged JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_jsonl_lines:
            f.write(line + '\n')
    
    print(f"🎯 BRUTAL MERGE COMPLETE!")
    print(f"📦 TOTAL LINES: {len(all_jsonl_lines)}")
    print(f"💾 OUTPUT: {output_file}")
    
    return output_file

if __name__ == "__main__":
    brutal_merge_all_data()