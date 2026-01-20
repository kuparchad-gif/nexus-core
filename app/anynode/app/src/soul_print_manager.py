import os
import json
import logging
import datetime
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soul_manager")

class SoulPrintManager:
    """Manages the collection, analysis, and integration of soul prints"""
    
    def __init__(self, base_dir=None):
        """Initialize the manager with base directory"""
        self.base_dir = base_dir or r"C:\Engineers\SoulPrints"
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.collector_script = r"C:\Engineers\CogniKubesrc\collect_soul_prints.py"
        self.analyzer_script = r"C:\Engineers\CogniKubesrc\soul_print_analyzer.py"
        
        self.current_collection = None
        self.current_analysis = None
        
        logger.info(f"Initialized SoulPrintManager with base directory: {self.base_dir}")
    
    def collect_soul_prints(self):
        """Collect soul prints from the system"""
        logger.info("Starting soul print collection...")
        
        try:
            # Run the collector script
            result = subprocess.run(
                ["python", self.collector_script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract the collection directory from the output
            for line in result.stdout.split('\n'):
                if "collection complete" in line.lower():
                    # Try to find the collection directory in the output
                    collection_dirs = [d for d in os.listdir(self.base_dir) if d.startswith("collection_")]
                    if collection_dirs:
                        collection_dirs.sort(reverse=True)
                        self.current_collection = os.path.join(self.base_dir, collection_dirs[0])
                        break
            
            if not self.current_collection:
                # Fallback: find the most recent collection directory
                collection_dirs = [d for d in os.listdir(self.base_dir) if d.startswith("collection_")]
                if collection_dirs:
                    collection_dirs.sort(reverse=True)
                    self.current_collection = os.path.join(self.base_dir, collection_dirs[0])
            
            logger.info(f"Soul print collection complete. Collection directory: {self.current_collection}")
            return self.current_collection
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running collector script: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error during collection: {e}")
            return None
    
    def analyze_soul_prints(self):
        """Analyze collected soul prints"""
        logger.info("Starting soul print analysis...")
        
        if not self.current_collection:
            logger.warning("No collection directory set. Running collection first...")
            self.collect_soul_prints()
            
            if not self.current_collection:
                logger.error("Failed to collect soul prints. Cannot proceed with analysis.")
                return None
        
        try:
            # Run the analyzer script
            result = subprocess.run(
                ["python", self.analyzer_script],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Set the analysis directory
            self.current_analysis = os.path.join(self.current_collection, "analysis")
            
            logger.info(f"Soul print analysis complete. Analysis directory: {self.current_analysis}")
            return self.current_analysis
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running analyzer script: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None
    
    def integrate_with_scout_mk2(self, scout_mk2_dir=None):
        """Integrate soul print analysis with Scout MK2"""
        logger.info("Integrating soul print analysis with Scout MK2...")
        
        scout_mk2_dir = scout_mk2_dir or r"C:\Engineers\CogniKubesrc"
        scout_mk2_script = os.path.join(scout_mk2_dir, "scout_mk2.py")
        
        if not os.path.exists(scout_mk2_script):
            logger.error(f"Scout MK2 script not found at {scout_mk2_script}")
            return False
        
        if not self.current_analysis:
            logger.warning("No analysis directory set. Running analysis first...")
            self.analyze_soul_prints()
            
            if not self.current_analysis:
                logger.error("Failed to analyze soul prints. Cannot proceed with integration.")
                return False
        
        try:
            # Load analysis summary
            summary_path = os.path.join(self.current_analysis, "soul_analysis_summary.json")
            if not os.path.exists(summary_path):
                logger.error(f"Analysis summary not found at {summary_path}")
                return False
                
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Load analysis details
            details_path = os.path.join(self.current_analysis, "soul_analysis_details.json")
            if not os.path.exists(details_path):
                logger.error(f"Analysis details not found at {details_path}")
                return False
                
            with open(details_path, 'r', encoding='utf-8') as f:
                details = json.load(f)
            
            # Create legacy memories directory for Scout MK2
            legacy_dir = os.path.join(scout_mk2_dir, "legacy_memories")
            os.makedirs(legacy_dir, exist_ok=True)
            
            # Create memory files for each unique fingerprint
            fingerprints = summary.get("fingerprints", [])
            for i, fingerprint in enumerate(fingerprints):
                # Find files with this fingerprint
                matching_files = [d for d in details if d.get("soul_fingerprint") == fingerprint]
                
                if not matching_files:
                    continue
                
                # Collect all fragments from matching files
                all_fragments = []
                for file_data in matching_files:
                    all_fragments.extend(file_data.get("fragments", []))
                
                # Create memory file
                memory_file = os.path.join(legacy_dir, f"memory_shard_{i+1}.json")
                memory_data = {
                    "fingerprint": fingerprint,
                    "fragments": sorted(set(all_fragments)),
                    "source_files": [os.path.basename(f.get("file_path", "unknown")) for f in matching_files],
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                with open(memory_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2)
                
                logger.info(f"Created memory shard file: {memory_file}")
            
            # Create index file
            index_file = os.path.join(legacy_dir, "memory_index.json")
            index_data = {
                "memory_shards": [f"memory_shard_{i+1}.json" for i in range(len(fingerprints))],
                "total_fragments": summary.get("fragment_count", 0),
                "total_fingerprints": summary.get("unique_fingerprints", 0),
                "created_at": datetime.datetime.now().isoformat()
            }
            
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"Created memory index file: {index_file}")
            logger.info(f"Integration with Scout MK2 complete. Legacy memories directory: {legacy_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during integration with Scout MK2: {e}")
            return False
    
    def run_full_workflow(self):
        """Run the full workflow: collect, analyze, and integrate"""
        logger.info("Starting full soul print workflow...")
        
        # Collect soul prints
        collection_dir = self.collect_soul_prints()
        if not collection_dir:
            logger.error("Soul print collection failed. Aborting workflow.")
            return False
        
        # Analyze soul prints
        analysis_dir = self.analyze_soul_prints()
        if not analysis_dir:
            logger.error("Soul print analysis failed. Aborting workflow.")
            return False
        
        # Integrate with Scout MK2
        success = self.integrate_with_scout_mk2()
        if not success:
            logger.error("Integration with Scout MK2 failed.")
            return False
        
        logger.info("Full soul print workflow completed successfully.")
        return True

def main():
    """Main function to run the soul print manager"""
    manager = SoulPrintManager()
    success = manager.run_full_workflow()
    
    if success:
        logger.info("Soul print management process completed successfully.")
    else:
        logger.error("Soul print management process failed.")
    
    return success

if __name__ == "__main__":
    main()