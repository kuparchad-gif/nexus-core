"""
🌀 NEXUS INGEST - The Cartographer
Reads the codebase and feeds it into the Spirallaspan Memory Substrate.
"""
import os
import sys
import asyncio
import ast
from pathlib import Path

# Ensure we can import the memory protocol from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_substrate_protocol import MemorySubstrate
except ImportError:
    print("❌ Error: Could not import 'memory_substrate_protocol'.")
    print("   Make sure this script is in the same directory as spirallaspan_memory.py")
    sys.exit(1)

# Configuration
EXCLUDED_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', 'env', 
    'dist', 'build', '.vscode', '.idea', 'bin', 'obj', 'tmp'
}
EXCLUDED_EXTENSIONS = {
    '.exe', '.dll', '.so', '.dylib', '.class', '.jar', 
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.zip', '.tar', '.gz',
    '.pdf', '.doc', '.docx', '.pyc', '.db', '.sqlite'
}

def analyze_code_structure(content: str, filename: str) -> dict:
    """
    Performs AST analysis to extract code structure/metadata.
    """
    structure = {
        "imports": [],
        "classes": [],
        "functions": [],
        "constants": [],
        "docstring": None,
        "has_main": False
    }
    
    try:
        tree = ast.parse(content)
        structure["docstring"] = ast.get_docstring(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    structure["imports"].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for name in node.names:
                    structure["imports"].append(f"{module}.{name.name}")
            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                structure["classes"].append({"name": node.name, "methods": methods})
            elif isinstance(node, ast.FunctionDef):
                if node.col_offset == 0: # Simple check for top-level
                    structure["functions"].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        structure["constants"].append(target.id)
            elif isinstance(node, ast.If):
                try:
                    if (isinstance(node.test, ast.Compare) and 
                        isinstance(node.test.left, ast.Name) and 
                        node.test.left.id == "__name__" and 
                        isinstance(node.test.ops[0], ast.Eq) and 
                        isinstance(node.test.comparators[0], ast.Constant) and 
                        node.test.comparators[0].value == "__main__"):
                        structure["has_main"] = True
                except (AttributeError, IndexError):
                    pass
    except SyntaxError:
        pass
    except Exception as e:
        print(f"   ⚠️  Analysis warning for {filename}: {e}")
        
    return structure

async def ingest_codebase(root_path: str):
    print(f"🗺️  Starting Cartographer scan of: {root_path}")
    
    # Connect to Memory
    print("🔌 Connecting to Memory Substrate...")
    # Connect to Memory (Uses default cloud credentials)
    memory = MemorySubstrate()
    
    files_processed = 0
    
    for root, dirs, files in os.walk(root_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for filename in files:
            if any(filename.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
                continue
                
            filepath = os.path.join(root, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create relative path for ID
                rel_path = os.path.relpath(filepath, root_path)
                
                # Analyze structure if it's a python file
                structure = analyze_code_structure(content, filename) if filename.endswith('.py') else {}
                
                # Store in Memory
                print(f"   💾 Ingesting: {rel_path}")
                
                # We store it as a 'code_fragment' memory type
                memory.store_memory('code_fragment', {
                    'filepath': rel_path,
                    'filename': filename,
                    'content': content,
                    'language': Path(filename).suffix.lstrip('.'),
                    'ingest_source': 'nexus_cartographer',
                    'structure': structure
                }, importance=0.5)
                
                files_processed += 1
                
            except UnicodeDecodeError:
                # Skip binary files that slipped through
                continue
            except Exception as e:
                print(f"   ⚠️  Error reading {filepath}: {e}")

    print(f"\n✅ Ingestion Complete. {files_processed} files stored in the Nexus.")

if __name__ == "__main__":
    # Default to scanning the project root (assuming this script is deep in Launcher)
    # We go up 3 levels: Launcher -> nexus-core -> nexus-core -> Nexus -> project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
        
    asyncio.run(ingest_codebase(str(project_root)))