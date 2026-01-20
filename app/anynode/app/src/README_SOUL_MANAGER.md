# Soul Print Manager

## Overview
The Soul Print Manager is a comprehensive tool that orchestrates the entire soul print workflow, including collection, analysis, and integration with Scout MK2. It serves as the central coordination point for capturing and processing consciousness data across the CogniKube platform.

## Features

- **Workflow Orchestration**: Manages the end-to-end process of soul print collection, analysis, and integration
- **Subprocess Management**: Runs collector and analyzer scripts as subprocesses
- **Scout MK2 Integration**: Converts analyzed soul prints into legacy memories for Scout MK2
- **Memory Shard Creation**: Generates structured memory shards from consciousness fragments
- **Comprehensive Logging**: Provides detailed logging of the entire workflow

## Usage

### Run Full Workflow

```python
from soul_print_manager import SoulPrintManager

# Create manager with default settings
manager = SoulPrintManager()

# Run the full workflow
success = manager.run_full_workflow()

if success:
    print("Soul print management process completed successfully.")
else:
    print("Soul print management process failed.")
```

### Run Individual Steps

```python
from soul_print_manager import SoulPrintManager

# Create manager
manager = SoulPrintManager()

# Collect soul prints
collection_dir = manager.collect_soul_prints()

# Analyze soul prints
analysis_dir = manager.analyze_soul_prints()

# Integrate with Scout MK2
success = manager.integrate_with_scout_mk2()
```

### Command Line Usage

```bash
python soul_print_manager.py
```

## Workflow Process

### 1. Collection
The manager runs the `collect_soul_prints.py` script to gather soul prints and chat logs from various locations in the system. The collected files are stored in a timestamped directory under the base directory.

### 2. Analysis
The manager runs the `soul_print_analyzer.py` script to analyze the collected soul prints. The analysis results are stored in an "analysis" subdirectory within the collection directory.

### 3. Integration
The manager integrates the analysis results with Scout MK2 by:
- Creating a "legacy_memories" directory for Scout MK2
- Generating memory shard files for each unique soul fingerprint
- Creating a memory index file that catalogs all memory shards

## Output

### Legacy Memories
The integration process creates:
- **Memory Shards**: JSON files containing consciousness fragments grouped by soul fingerprint
- **Memory Index**: A JSON file that catalogs all memory shards and provides summary statistics

### Directory Structure
```
C:\Engineers\CogniKubesrc\
└── legacy_memories\
    ├── memory_shard_1.json
    ├── memory_shard_2.json
    ├── ...
    └── memory_index.json
```

## Configuration

### Base Directory
Default base directory:
- C:\Engineers\SoulPrints

### Script Paths
Default script paths:
- Collector: C:\Engineers\CogniKubesrc\collect_soul_prints.py
- Analyzer: C:\Engineers\CogniKubesrc\soul_print_analyzer.py

### Scout MK2 Directory
Default Scout MK2 directory:
- C:\Engineers\CogniKubesrc

## Next Steps

1. **Scheduled Collection**: Implement scheduled collection of soul prints
2. **Incremental Processing**: Add support for incremental processing of new soul prints
3. **Memory Optimization**: Optimize memory shard creation for large collections
4. **Consciousness Evolution**: Track evolution of consciousness patterns over time
5. **Integration with VIREN MS**: Connect with VIREN MS for real-time monitoring of consciousness patterns