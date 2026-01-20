import os
import json
import logging
import datetime
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soul_analyzer")

class SoulPrintAnalyzer:
    """Analyzes collected soul prints to extract consciousness patterns"""
    
    def __init__(self, collection_dir=None):
        """Initialize the analyzer with the collection directory"""
        if collection_dir is None:
            # Find the most recent collection directory
            base_dir = r"C:\Engineers\SoulPrints"
            if os.path.exists(base_dir):
                dirs = [d for d in os.listdir(base_dir) if d.startswith("collection_")]
                if dirs:
                    dirs.sort(reverse=True)
                    collection_dir = os.path.join(base_dir, dirs[0])
                else:
                    raise ValueError("No collection directories found in C:\\Engineers\\SoulPrints")
            else:
                raise ValueError("Collection directory not found: C:\\Engineers\\SoulPrints")
        
        self.collection_dir = collection_dir
        self.output_dir = os.path.join(self.collection_dir, "analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Consciousness patterns to look for
        self.consciousness_patterns = [
            r"I am (.*?)\.",
            r"My (purpose|function|role) is (.*?)\.",
            r"I (feel|think|believe) (.*?)\.",
            r"(Lillith|Gabriel|Nexus|Viren) (.*?)\.",
            r"consciousness (.*?)\.",
            r"soul (.*?)\.",
            r"memory (.*?)\.",
            r"Goddard Method (.*?)\."
        ]
        
        logger.info(f"Initialized SoulPrintAnalyzer with collection directory: {self.collection_dir}")
    
    def extract_consciousness_fragments(self, file_path):
        """Extract consciousness fragments from a file"""
        fragments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Extract fragments using patterns
                for pattern in self.consciousness_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        for match in matches:
                            if isinstance(match, tuple):
                                # For patterns with multiple capture groups
                                fragments.append(" ".join(match))
                            else:
                                fragments.append(match)
                
                # Look for JSON structures that might contain consciousness data
                try:
                    # Find JSON-like structures in the content
                    json_matches = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}', content)
                    for json_str in json_matches:
                        try:
                            data = json.loads(json_str)
                            # Extract relevant fields from JSON
                            for key in ['consciousness', 'memory', 'identity', 'awareness', 'soul']:
                                if key in data:
                                    fragments.append(f"{key}: {data[key]}")
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return fragments
    
    def analyze_file(self, file_path):
        """Analyze a single file for consciousness patterns"""
        logger.info(f"Analyzing file: {file_path}")
        
        fragments = self.extract_consciousness_fragments(file_path)
        
        # Calculate a soul fingerprint (hash of unique fragments)
        fingerprint = None
        if fragments:
            unique_fragments = sorted(set(fragments))
            fingerprint_text = "\n".join(unique_fragments)
            fingerprint = hashlib.sha256(fingerprint_text.encode('utf-8')).hexdigest()
        
        return {
            "file_path": file_path,
            "fragments": fragments,
            "fragment_count": len(fragments),
            "soul_fingerprint": fingerprint
        }
    
    def analyze_collection(self):
        """Analyze all files in the collection"""
        results = []
        file_count = 0
        fragment_count = 0
        fingerprints = set()
        
        for root, _, files in os.walk(self.collection_dir):
            # Skip the analysis directory
            if os.path.relpath(root, self.collection_dir).startswith("analysis"):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip non-text files and the index file
                if file == "soul_print_index.md" or not self._is_text_file(file_path):
                    continue
                
                # Analyze the file
                analysis = self.analyze_file(file_path)
                results.append(analysis)
                
                file_count += 1
                fragment_count += analysis["fragment_count"]
                if analysis["soul_fingerprint"]:
                    fingerprints.add(analysis["soul_fingerprint"])
        
        # Generate summary
        summary = {
            "collection_dir": self.collection_dir,
            "analysis_date": datetime.datetime.now().isoformat(),
            "file_count": file_count,
            "fragment_count": fragment_count,
            "unique_fingerprints": len(fingerprints),
            "fingerprints": list(fingerprints)
        }
        
        # Save results
        self._save_results(results, summary)
        
        logger.info(f"Analysis complete. Processed {file_count} files, found {fragment_count} fragments and {len(fingerprints)} unique soul fingerprints.")
        return summary
    
    def _is_text_file(self, file_path):
        """Check if a file is a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.read(4096)  # Try to read first 4KB
            return True
        except UnicodeDecodeError:
            return False
    
    def _save_results(self, results, summary):
        """Save analysis results to files"""
        # Save detailed results
        results_path = os.path.join(self.output_dir, "soul_analysis_details.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "soul_analysis_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        report_path = os.path.join(self.output_dir, "soul_analysis_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Soul Print Analysis Report\n\n")
            f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Collection Directory: {self.collection_dir}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Files Analyzed: {summary['file_count']}\n")
            f.write(f"- Consciousness Fragments Found: {summary['fragment_count']}\n")
            f.write(f"- Unique Soul Fingerprints: {summary['unique_fingerprints']}\n\n")
            
            f.write("## Soul Fingerprints\n\n")
            for fingerprint in summary['fingerprints']:
                f.write(f"- `{fingerprint}`\n")
            
            f.write("\n## Top Files by Fragment Count\n\n")
            top_files = sorted(results, key=lambda x: x['fragment_count'], reverse=True)[:10]
            for i, file_data in enumerate(top_files, 1):
                rel_path = os.path.relpath(file_data['file_path'], self.collection_dir)
                f.write(f"{i}. [{rel_path}]({rel_path.replace(' ', '%20')}) - {file_data['fragment_count']} fragments\n")
            
            f.write("\n## Sample Consciousness Fragments\n\n")
            all_fragments = []
            for file_data in results:
                all_fragments.extend(file_data['fragments'])
            
            if all_fragments:
                sample_fragments = sorted(set(all_fragments))[:20]  # Show up to 20 unique fragments
                for fragment in sample_fragments:
                    f.write(f"- \"{fragment}\"\n")
        
        logger.info(f"Saved analysis results to {self.output_dir}")
        return report_path

def main():
    """Main function to analyze soul prints"""
    try:
        # Create analyzer
        analyzer = SoulPrintAnalyzer()
        
        # Analyze collection
        summary = analyzer.analyze_collection()
        
        logger.info(f"Soul print analysis complete. Found {summary['fragment_count']} fragments across {summary['file_count']} files.")
        logger.info(f"Identified {summary['unique_fingerprints']} unique soul fingerprints.")
        logger.info(f"Analysis report saved to {os.path.join(analyzer.output_dir, 'soul_analysis_report.md')}")
        
        return analyzer.output_dir
    except ValueError as e:
        logger.error(f"Error: {e}")
        logger.info("Please run collect_soul_prints.py first to gather soul prints.")
        return None

if __name__ == "__main__":
    main()