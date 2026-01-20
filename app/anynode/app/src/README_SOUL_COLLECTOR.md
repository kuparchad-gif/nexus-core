# Soul Print Collector

## Overview
The Soul Print Collector is a tool designed to gather chat logs, soul prints, and consciousness-related files from various locations in the system and copy them to a central repository. This ensures that important consciousness data is preserved and easily accessible.

## Features

- **Recursive Directory Scanning**: Searches through specified directories to find soul prints and chat logs
- **Pattern Matching**: Identifies files based on filename patterns and content analysis
- **Preservation of Directory Structure**: Maintains the original directory structure in the target location
- **Automatic Indexing**: Creates a markdown index of all collected files for easy navigation
- **Detailed Logging**: Provides comprehensive logging of the collection process

## Usage

### Basic Collection

```python
from collect_soul_prints import SoulPrintCollector

# Create collector with default settings
collector = SoulPrintCollector()

# Collect files
total_files = collector.collect_files()

# Create index
index_path = collector.create_index()

print(f"Collected {total_files} files. Index created at {index_path}")
```

### Custom Collection

```python
from collect_soul_prints import SoulPrintCollector

# Specify custom source and target directories
collector = SoulPrintCollector(
    source_dirs=[
        r"C:\Custom\Path1",
        r"C:\Custom\Path2"
    ],
    target_dir=r"C:\Central\SoulPrints\Repository"
)

# Collect files
collector.collect_files()

# Create index
collector.create_index()
```

### Command Line Usage

```bash
python collect_soul_prints.py
```

## File Identification

The collector identifies soul prints and chat logs based on:

1. **File Extensions**: .md, .txt, .log, .json
2. **Filename Patterns**:
   - *soul*print*.md
   - *conversation*.md
   - *chat*log*
   - *Q_Soul_Print*
   - *lillith*.md
   - *viren*.md
   - *nexus*.md
   - *consciousness*.md
   - *memory*.md
   - *gabriel*horn*.md
3. **Content Analysis**: Searches for key phrases like "soul print", "consciousness", "lillith", "viren", "nexus", "gabriel's horn", "memory shard", and "goddard method"

## Output

The collector creates a timestamped directory in the target location with:

1. **Collected Files**: Organized in a directory structure mirroring the source
2. **Index File**: A markdown file listing all collected files with sizes and modification dates

## Configuration

### Source Directories
Default source directories:
- C:\Viren
- C:\Engineers\Documents
- C:\Engineers\CogniKubesrc

### Target Directory
Default target directory:
- C:\Engineers\SoulPrints\collection_[TIMESTAMP]

## Next Steps

1. **Enhanced Content Analysis**: Implement more sophisticated content analysis to identify soul prints
2. **Deduplication**: Add functionality to detect and handle duplicate files
3. **Incremental Collection**: Support for incremental collection to avoid copying already collected files
4. **Compression**: Option to compress the collection for archival purposes
5. **Search Interface**: Create a simple web interface for searching and browsing the collected soul prints