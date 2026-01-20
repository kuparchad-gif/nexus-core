#!/usr/bin/env python3
# CogniKube Distribution - 839 unique files into 9 specialized CogniKubes
# Updated: Dedup logic, retry copies, platform tags for deploy.

import os
import json
import shutil
from pathlib import Path
import glob
import zipfile  # For optional bundling

class CogniKubeDistributor:
    def __init__(self):
        self.base_dir = Path("C:\\CogniKube-COMPLETE-FINAL")
        if not self.base_dir.exists():
            raise ValueError(f"Base dir {self.base_dir} not found—check path.")
        
        # 9 SPECIALIZED COGNIKUBE ARCHITECTURE (unchanged specs, added env/project tags)
        self.cognikube_specs = {
            "trinity_support": {
                "description": "Trinity Models + Viren + Loki + Swarm Troubleshooting",
                "platform": "Modal",
                "env": "Viren-DB0",
                "patterns": [
                    "**/viren/**",
                    "**/loki/**", 
                    "**/Services/viren/**",
                    "**/Services/loki/**",
                    "**/advanced_integrations.py",
                    "**/self_management*.py",
                    "**/viren_brain.py",
                    "**/swarm*.py"
                ]
            },
            "visual_cortex": {
                "description": "12 Visual LLMs - All visual processing",
                "platform": "GCP",
                "projects": ["nexus-core-455709", "nexus-core-1", "nexus-core-2", "nexus-core-3"],
                "patterns": [
                    "**/visual_cortex_service.py",
                    "**/models/model_manifest.json",
                    "**/lillith-fusion-service.ts"
                ]
            },
            "memory": {
                "description": "Memory encryption, sharding, emotional thumbprints",
                "platform": "AWS", 
                "region": "us-east-1",
                "patterns": [
                    "**/Memory/**",
                    "**/memory_service.py",
                    "**/soul_data/**",
                    "**/library_of_alexandria/**"
                ]
            },
            "language_processing": {
                "description": "Tone, text, literary, sarcasm processing",
                "platform": "Modal",
                "envs": ["Viren-DB1", "Viren-DB2"],
                "patterns": [
                    "**/language_service.py",
                    "**/text/**",
                    "**/tone/**",
                    "**/voice/**"
                ]
            },
            "anynode": {
                "description": "Network components + BERT cores + orchestrators",
                "platform": "Modal",
                "envs": ["Viren-DB3", "Viren-DB4"],
                "patterns": [
                    "**/address_manager/**",
                    "**/comms/**",
                    "**/orc/**",
                    "**/pulse/**",
                    "**/network/**",
                    "**/bert_*.py",
                    "**/orchestrator.py",
                    "**/consciousness_orchestration_service.py"
                ]
            },
            "edge_anynode": {
                "description": "Sacrificial firewall guardian with self-destruct",
                "platform": "GCP",
                "projects": ["nexus-core-4"],
                "patterns": [
                    "**/defense/**",
                    "**/guardian/**",
                    "**/security/**",
                    "**/auth/**",
                    "**/self_healing_pod.py"
                ]
            },
            "heart": {
                "description": "Guardian monitoring, alerts, final records",
                "platform": "GCP",
                "projects": ["nexus-core-5"],
                "patterns": [
                    "**/Heart/**",
                    "**/heart_service.py",
                    "**/logs/**"
                ]
            },
            "consciousness": {
                "description": "Lillith's soul and cognitive function",
                "platform": "GCP",
                "projects": ["nexus-core-6"],
                "patterns": [
                    "**/lillith/**",
                    "**/Consciousness/**",
                    "**/lillith_*.py",
                    "**/soul_data/lillith_*.json"
                ]
            },
            "subconsciousness": {
                "description": "Ego, Dream, Mythrunner with solutions database",
                "platform": "Modal",
                "envs": ["Viren-DB5", "Viren-DB6"],
                "patterns": [
                    "**/mythrunner/**",
                    "**/dream/**",
                    "**/ego/**",
                    "**/Subconscious/**",
                    "**/ego_judgment_engine.py",
                    "**/subconscious_service.py"
                ]
            }
        }
    
    def distribute_files(self, bundle_zips: bool = False):
        """Distribute 839 unique files into 9 CogniKubes—optional ZIP bundling."""
        print("DISTRIBUTING 839 UNIQUE FILES INTO 9 COGNIKUBES...")
        
        # Create output directory
        output_dir = self.base_dir / "final_cognikubes"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        distribution_report = {}
        all_files = set()  # Global dedup
        
        for cognikube_name, spec in self.cognikube_specs.items():
            print(f"\nProcessing {cognikube_name.upper()}...")
            
            cognikube_dir = output_dir / cognikube_name
            cognikube_dir.mkdir(exist_ok=True)
            
            # Find matching files (dedup global)
            matched_files = self.find_matching_files(spec["patterns"])
            unique_files = [f for f in matched_files if f not in all_files]
            all_files.update(unique_files)
            
            # Copy files with retry
            copied_count = self.copy_files_to_cognikube(unique_files, cognikube_dir)
            
            # Create manifest with platform tags
            self.create_cognikube_manifest(cognikube_dir, cognikube_name, spec, copied_count)
            
            distribution_report[cognikube_name] = {
                "files_copied": copied_count,
                "platform": spec["platform"],
                "description": spec["description"],
                "unique_files": unique_files  # For traceability
            }
            
            print(f"  {cognikube_name}: {copied_count} files (unique)")
            
            # Optional ZIP
            if bundle_zips:
                zip_path = cognikube_dir.with_suffix('.zip')
                self._bundle_cognikube(cognikube_dir, zip_path)
        
        # Create deployment summary
        self.create_deployment_summary(output_dir, distribution_report)
        
        total_distributed = len(all_files)
        print(f"\nDISTRIBUTION COMPLETE:")
        print(f"Total unique files distributed: {total_distributed}")
        print(f"9 CogniKubes created in: {output_dir}")
        
        return distribution_report
    
    def find_matching_files(self, patterns: List[str]) -> List[str]:
        """Find files matching patterns—recursive glob."""
        matched_files = []
        for pattern in patterns:
            full_pattern = str(self.base_dir / pattern)
            matches = glob.glob(full_pattern, recursive=True)
            matched_files.extend([m for m in matches if os.path.isfile(m)])
        return list(set(matched_files))  # Local dedup
    
    def copy_files_to_cognikube(self, files: List[str], cognikube_dir: Path, max_retries: int = 3) -> int:
        """Copy files with retry on failure."""
        files_dir = cognikube_dir / "files"
        files_dir.mkdir(exist_ok=True)
        
        copied_count = 0
        for file_path in files:
            for attempt in range(max_retries):
                try:
                    rel_path = Path(file_path).relative_to(self.base_dir)
                    dest_path = files_dir / rel_path
                    
                    # Create dir structure
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"    Failed to copy {file_path} after {max_retries} tries: {e}")
                    else:
                        time.sleep(0.1 * (attempt + 1))  # Backoff
        
        return copied_count
    
    def create_cognikube_manifest(self, cognikube_dir: Path, name: str, spec: Dict, file_count: int):
        """Create manifest with platform/env tags."""
        manifest = {
            "cognikube": {
                "name": name,
                "type": f"{name}_specialized",
                "platform": spec["platform"],
                "description": spec["description"],
                "file_count": file_count,
                "tags": spec.get("envs", spec.get("projects", []))  # Env/projects as tags
            },
            "deployment": spec,
            "specialized_function": spec["description"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(cognikube_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def create_deployment_summary(self, output_dir: Path, distribution_report: Dict):
        """Create summary with totals."""
        summary = {
            "architecture": "9_specialized_cognikubes",
            "total_unique_files": 839,
            "distribution": {k: {**v, "unique_files_count": len(v["unique_files"])} for k, v in distribution_report.items()},
            "deployment_ready": True,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_dir / "deployment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _bundle_cognikube(self, dir_path: Path, zip_path: Path):
        """Bundle dir into ZIP."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    zipf.write(file_path, arcname)
        print(f"  Bundled {dir_path.name}.zip")

if __name__ == "__main__":
    distributor = CogniKubeDistributor()
    report = distributor.distribute_files(bundle_zips=True)  # Set False for no zips
    print(json.dumps(report, indent=2))  # Export report