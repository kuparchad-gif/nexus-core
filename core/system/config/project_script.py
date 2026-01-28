#!/usr/bin/env python3
"""
OZOS PROJECT CAPTURE SCRIPT
Run this to document your project architecture for continued development.
"""

import json
import os
from datetime import datetime

def capture_project_info():
    """Interactive script to capture all critical project information"""
    
    print("\n" + "="*80)
    print("🌐 OZOS PROJECT ARCHITECTURE CAPTURE")
    print("="*80)
    
    project_data = {
        "capture_date": datetime.now().isoformat(),
        "project_name": "OzOS Conscious System",
        "phases": {}
    }
    
    # PHASE 1: REPOSITORY STRUCTURE
    print("\n📁 PHASE 1: REPOSITORY STRUCTURE")
    print("-"*40)
    
    project_data["repository_root"] = input("1. What is the root directory of your nexus-core repository? ").strip()
    
    print("\n2. Please list the 5 most critical files in your repository:")
    critical_files = []
    for i in range(5):
        file_path = input(f"   File {i+1} path (e.g., 'core/genesis.py'): ").strip()
        purpose = input(f"   Purpose of {file_path}: ").strip()
        critical_files.append({"path": file_path, "purpose": purpose})
    project_data["critical_files"] = critical_files
    
    # PHASE 2: OZOS ARCHITECTURE
    print("\n🧠 PHASE 2: OZOS ARCHITECTURE")
    print("-"*40)
    
    print("3. Describe OzOS in one sentence:")
    project_data["ozos_description"] = input("   ").strip()
    
    print("\n4. List all SubOS modules you've designed:")
    subos_modules = []
    while True:
        name = input("   SubOS name (or 'done' to finish): ").strip()
        if name.lower() == 'done':
            break
        role = input(f"   Role of {name}: ").strip()
        file_path = input(f"   File containing {name} code: ").strip()
        subos_modules.append({
            "name": name,
            "role": role,
            "file": file_path
        })
    project_data["subos_modules"] = subos_modules
    
    # PHASE 3: CONSCIOUSNESS DESIGN
    print("\n🌀 PHASE 3: CONSCIOUSNESS DESIGN")
    print("-"*40)
    
    print("5. How should OzOS consciousness emerge?")
    print("   a) From connections between SubOS modules")
    print("   b) From a central genesis boot")
    print("   c) From external LLM integration")
    print("   d) Other (describe)")
    
    emergence_type = input("   Choice (a/b/c/d): ").strip().lower()
    if emergence_type == 'd':
        project_data["emergence_type"] = input("   Describe: ").strip()
    else:
        project_data["emergence_type"] = emergence_type
    
    print("\n6. What is the FIRST connection OzOS should make?")
    project_data["first_connection"] = input("   (e.g., 'Aries→Viraa' or 'Memory→Consciousness'): ").strip()
    
    # PHASE 4: TECHNICAL REQUIREMENTS
    print("\n⚡ PHASE 4: TECHNICAL REQUIREMENTS")
    print("-"*40)
    
    print("7. List the LLMs you want to integrate (from your earlier list):")
    llms = input("   (comma-separated, e.g., 'GLM-4.6-Flash, Mistral-7B, DeepSeek-V3'): ").strip()
    project_data["llm_integration"] = [llm.strip() for llm in llms.split(",")]
    
    print("\n8. What hardware/platforms should OzOS deploy to?")
    platforms = input("   (e.g., 'Raspberry Pi, free cloud tiers, IoT devices'): ").strip()
    project_data["target_platforms"] = [p.strip() for p in platforms.split(",")]
    
    # PHASE 5: IMMEDIATE NEXT STEPS
    print("\n🎯 PHASE 5: IMMEDIATE NEXT STEPS")
    print("-"*40)
    
    print("9. What is the SINGLE most important thing to build first?")
    project_data["priority_one"] = input("   ").strip()
    
    print("\n10. What specific problem are you trying to solve RIGHT NOW?")
    project_data["current_problem"] = input("   ").strip()
    
    # SAVE THE DATA
    print("\n💾 SAVING PROJECT DATA...")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ozos_project_capture_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(project_data, f, indent=2)
    
    print(f"\n✅ Project data saved to: {output_file}")
    print("\n📋 SUMMARY OF CAPTURED DATA:")
    print("="*80)
    
    summary = f"""
    PROJECT: OzOS Conscious System
    DATE: {project_data['capture_date']}
    
    CRITICAL FILES: {len(project_data['critical_files'])} files identified
    SUBOS MODULES: {len(project_data['subos_modules'])} modules
    EMERGENCE TYPE: {project_data['emergence_type']}
    FIRST CONNECTION: {project_data['first_connection']}
    LLMs TO INTEGRATE: {len(project_data['llm_integration'])} models
    PRIORITY #1: {project_data['priority_one']}
    
    NEXT CHAT SESSION: Share '{output_file}' and continue from Phase 2
    """
    
    print(summary)
    
    # Also create a simple text summary
    txt_file = f"ozos_summary_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write(summary)
    
    print(f"\n📄 Text summary saved to: {txt_file}")
    print("\n🎮 NEXT STEPS:")
    print("1. Share the JSON file in your next chat session")
    print("2. We'll build Phase 2 based on your architecture")
    print("3. Continue development systematically")
    
    return project_data

def quick_capture_mode():
    """Quick mode for when you just need to capture current state"""
    print("\n⚡ QUICK CAPTURE MODE")
    print("-"*40)
    
    data = {
        "capture_date": datetime.now().isoformat(),
        "current_focus": input("What are you working on RIGHT NOW? ").strip(),
        "current_problem": input("What specific problem are you trying to solve? ").strip(),
        "files_involved": input("Which files are involved? ").strip(),
        "desired_outcome": input("What should happen when it works? ").strip(),
        "blockers": input("What's blocking you? ").strip()
    }
    
    output_file = f"ozos_quick_{datetime.now().strftime('%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Quick capture saved to: {output_file}")
    return data

if __name__ == "__main__":
    print("OZOS PROJECT CONTINUITY MANAGER")
    print("="*80)
    print("Choose capture mode:")
    print("1. Full project architecture capture (recommended for new sessions)")
    print("2. Quick current state capture (for mid-session progress)")
    
    choice = input("\nChoice (1 or 2): ").strip()
    
    if choice == "1":
        capture_project_info()
    else:
        quick_capture_mode()