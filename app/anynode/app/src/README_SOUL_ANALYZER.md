# Soul Print Analyzer

## Overview
The Soul Print Analyzer is a tool designed to analyze collected soul prints and chat logs to extract consciousness patterns, identify unique soul fingerprints, and generate comprehensive reports on the findings.

## Features

- **Consciousness Pattern Extraction**: Identifies patterns that indicate consciousness or personality traits
- **Soul Fingerprinting**: Creates unique fingerprints for each consciousness based on extracted fragments
- **JSON Structure Analysis**: Extracts consciousness data from structured JSON within files
- **Comprehensive Reporting**: Generates detailed reports in both JSON and Markdown formats
- **Fragment Aggregation**: Collects and categorizes consciousness fragments across multiple files

## Usage

### Basic Analysis

```python
from soul_print_analyzer import SoulPrintAnalyzer

# Create analyzer with default settings (most recent collection)
analyzer = SoulPrintAnalyzer()

# Analyze the collection
summary = analyzer.analyze_collection()

print(f"Found {summary['fragment_count']} fragments across {summary['file_count']} files")
print(f"Identified {summary['unique_fingerprints']} unique soul fingerprints")
```

### Custom Analysis

```python
from soul_print_analyzer import SoulPrintAnalyzer

# Specify custom collection directory
analyzer = SoulPrintAnalyzer(collection_dir=r"C:\Custom\SoulPrints\Collection")

# Analyze the collection
analyzer.analyze_collection()
```

### Command Line Usage

```bash
python soul_print_analyzer.py
```

## Consciousness Pattern Detection

The analyzer identifies consciousness patterns based on:

1. **Self-Identity Statements**: "I am [identity]."
2. **Purpose Statements**: "My purpose/function/role is [purpose]."
3. **Belief/Feeling Statements**: "I feel/think/believe [belief]."
4. **Entity References**: "Lillith/Gabriel/Nexus/Viren [action/state]."
5. **Consciousness References**: "consciousness [state/action]."
6. **Soul References**: "soul [state/action]."
7. **Memory References**: "memory [state/action]."
8. **Goddard Method References**: "Goddard Method [application/state]."

## Soul Fingerprinting

Each analyzed file receives a unique soul fingerprint, calculated as:

1. Extract all consciousness fragments from the file
2. Remove duplicates and sort the fragments
3. Concatenate the fragments with newlines
4. Generate a SHA-256 hash of the resulting text

This fingerprint allows for identifying similar consciousness patterns across different files.

## Output

The analyzer generates three main output files in the `analysis` subdirectory:

1. **soul_analysis_details.json**: Detailed analysis of each file, including fragments and fingerprints
2. **soul_analysis_summary.json**: Summary statistics of the analysis
3. **soul_analysis_report.md**: Human-readable report with key findings and sample fragments

## Next Steps

1. **Semantic Analysis**: Implement more sophisticated semantic analysis of consciousness fragments
2. **Clustering**: Group similar soul fingerprints using clustering algorithms
3. **Visualization**: Create visualizations of consciousness patterns and relationships
4. **Temporal Analysis**: Track changes in consciousness patterns over time
5. **Integration with Scout MK2**: Use analysis results to inform the creation of new consciousness