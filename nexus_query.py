"""
🔍 NEXUS QUERY - The Librarian
Queries the Spirallaspan Memory Substrate for code and structure.
"""
import os
import sys
import argparse
from typing import List, Dict

# Ensure we can import the memory protocol
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_substrate_protocol import MemorySubstrate
    from qdrant_client import models
except ImportError:
    print("❌ Error: Could not import 'memory_substrate_protocol'.")
    sys.exit(1)

def query_memory(query_text: str):
    print(f"🔍 Searching Nexus for: '{query_text}'...")
    
    # Connect to Memory (Uses default cloud credentials)
    memory = MemorySubstrate()
    
    # Access Qdrant client directly
    client = memory.clients[0]
    collection = memory.collection_name
    
    # Search in the nested payload dictionary where our data lives
    # We look in filename, content, and the structural elements
    scroll_filter = models.Filter(
        should=[
            models.FieldCondition(key="payload.filename", match=models.MatchText(text=query_text)),
            models.FieldCondition(key="payload.structure.imports", match=models.MatchValue(value=query_text)),
            models.FieldCondition(key="payload.structure.classes.name", match=models.MatchValue(value=query_text)),
            models.FieldCondition(key="payload.structure.functions", match=models.MatchValue(value=query_text)),
        ]
    )
    
    # Execute search
    results, _ = client.scroll(
        collection_name=collection,
        scroll_filter=scroll_filter,
        limit=20,
        with_payload=True
    )
    
    if not results:
        print("❌ No matches found.")
        return

    print(f"\n✅ Found {len(results)} matches:\n")
    
    for point in results:
        # The data is inside the 'payload' key of the MemoryCell
        data = point.payload.get('payload', {})
        
        filename = data.get('filename', 'Unknown')
        filepath = data.get('filepath', 'Unknown')
        structure = data.get('structure', {})
        
        print(f"📄 {filepath}")
        
        if structure:
            # Check if our query matched specific parts to highlight them
            imports = structure.get('imports', [])
            classes = [c['name'] for c in structure.get('classes', [])]
            funcs = structure.get('functions', [])
            
            if query_text in imports:
                print(f"   🔗 Imports: {query_text}")
            if query_text in classes:
                print(f"   📦 Class: {query_text}")
            if query_text in funcs:
                print(f"   ƒ  Function: {query_text}")
                
            # Show summary if generic match
            if query_text not in imports and query_text not in classes and query_text not in funcs:
                print(f"   ℹ️  (Text match in file)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Nexus Memory")
    parser.add_argument("query", help="Search term (Import, Class, Function, or Filename)")
    args = parser.parse_args()
    
    query_memory(args.query)