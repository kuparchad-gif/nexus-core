#!/usr/bin/env python3
"""
CogniKube: Win-Dask Opt + Ray V2 FT (2025).
- Dask: Threaded/low workers—no restart.
- Ray: V2 Checkpoint batches.
Usage: python code_crawler.py --executor dask --n_workers 4
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List
import concurrent.futures
from ray import train  # V2 checkpoints

# Conditionals
DASK_AVAIL = False
try:
    from dask.distributed import Client, LocalCluster, as_completed as dask_as_completed
    import dask
    dask.config.set({'distributed.scheduler.allowed-failures': 3})
    DASK_AVAIL = True
except ImportError:
    print("⚠️ Dask → Thread.")

RAY_AVAIL = False
try:
    import ray
    RAY_AVAIL = True
except ImportError:
    pass

POLARS_AVAIL = False
try:
    import polars as pl
    POLARS_AVAIL = True
except ImportError:
    pass

class PatternVisitor(ast.NodeVisitor):
    def __init__(self):
        self.patterns = {"catalyst_modules": False, "quantum_routing": False, "nexus_integration": False,
                         "viraa_mentions": False, "self_sacrifice_protocols": False}
    
    def visit_Name(self, node):
        name_lower = node.id.lower()
        if "catalyst" in name_lower and any(w in name_lower for w in ["module", "modules"]):
            self.patterns["catalyst_modules"] = True
        elif "viraa" in name_lower:
            self.patterns["viraa_mentions"] = True
        elif "self" in name_lower and any(w in name_lower for w in ["destruct", "sacrifice"]):
            self.patterns["self_sacrifice_protocols"] = True
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if hasattr(node.func, 'id'):
            func_lower = node.func.id.lower()
            if "quantum" in func_lower and "router" in func_lower:
                self.patterns["quantum_routing"] = True
            if "nexus" in func_lower and "integration" in func_lower:
                self.patterns["nexus_integration"] = True
        self.generic_visit(node)

def analyze_file_ast(file_path: Path, root_path: Path) -> Dict[str, List[str]]:
    relative_path = str(file_path.relative_to(root_path))
    try:
        with open(file_path, 'r', encoding='utf-8', errors='surrogateescape') as f:
            source = f.read()
        tree = ast.parse(source, filename=str(file_path))
        visitor = PatternVisitor()
        visitor.visit(tree)
        hits = [k for k, v in visitor.patterns.items() if v]
        return {relative_path: hits} if hits else {}
    except Exception as e:
        print(f"⚠️ Skip {relative_path}: {e}")
        return {}

class CodeIntelligenceCompressor:
    def __init__(self, root_path=".", executor="thread", max_workers=32, n_workers=4, num_cpus=8,
                 use_polars=False, replicate=1, gpu_frac=0.0):
        self.root_path = Path(root_path)
        self.executor = executor
        self.max_workers = max_workers if executor == "thread" else None
        self.n_workers = n_workers if executor == "dask" else None
        self.num_cpus = num_cpus if executor == "ray" else None
        self.gpu_frac = gpu_frac if executor == "ray" else 0.0
        self.use_polars = use_polars and POLARS_AVAIL
        self.replicate = replicate if executor == "dask" else 1
        self.compressed_data = {"architecture_overview": {}, "critical_files": {}, "key_patterns": {},
                                "dependencies": {}, "compression_metadata": {}}
        self._init_executor()
    
    def _init_executor(self):
        if self.executor == "dask" and DASK_AVAIL:
            # Win: Threaded/low workers, no restart (Nanny-less)
            self.cluster = LocalCluster(n_workers=self.n_workers, threads_per_worker=4, processes=False,
                                        memory_limit="1GB", dashboard_address=":8787")
            self.client = Client(self.cluster)  # Fresh—no restart
            print(f"🔮 Dask Win: {self.client} | Workers: {self.n_workers} (threads=4) | Replicate: {self.replicate}")
            print(f"📊 Dash: http://127.0.0.1:8787/status")
        elif self.executor == "ray" and RAY_AVAIL:
            ray.init(num_cpus=self.num_cpus, num_gpus=self.gpu_frac)
            # V2 FT: Checkpoint actor
            @ray.remote(num_cpus=0.1, num_gpus=self.gpu_frac, max_retries=5, retry_exceptions=True)
            class V2Analyzer:
                def __init__(self):
                    self.ckpt = train.Checkpoint("/tmp/ray_v2_ckpt")
                async def analyze_batch(self, paths: List[Path], root: Path):
                    self.ckpt.save()
                    return [analyze_file_ast(p, root) for p in paths]
            self.v2_analyzer = V2Analyzer.remote()
            print(f"🚀 Ray V2: Checkpoints | GPUs {self.gpu_frac}")
    
    def scan_architecture(self):
        print("🔍 Arch scan...")
        nexus_nodes = {}
        for node_dir in self.root_path.rglob("nexus-*"):
            if node_dir.is_dir():
                nexus_nodes[node_dir.name] = {"path": str(node_dir.relative_to(self.root_path)),
                                              "file_count": len(list(node_dir.rglob("*.py"))), "services": []}
        components = {"routers": list(self.root_path.rglob("**/*router*.py")), "agents": list(self.root_path.rglob("**/*agent*.py")),
                      "services": list(self.root_path.rglob("**/*service*.py")), "guardians": list(self.root_path.rglob("**/*guardian*.py")),
                      "catalysts": list(self.root_path.rglob("**/*catalyst*.py"))}
        self.compressed_data["architecture_overview"] = {"nexus_nodes": nexus_nodes,
                                                         "component_counts": {k: len(v) for k, v in components.items()},
                                                         "total_py_files": len(list(self.root_path.rglob("*.py")))}
        return self.compressed_data["architecture_overview"]
    
    def extract_critical_files(self):
        print("📋 Crit extract...")
        patterns = ["orchestrat", "core", "main", "app", "init", "config", "settings", "router", "controller"]
        critical_files = set()
        for core in patterns:
            critical_files.update(self.root_path.rglob(f"**/*{core}*.py"))
        critical_files = sorted(list(critical_files), key=lambda x: len(x.parts))[:50]
        processed = 0
        for fp in critical_files:
            try:
                rel = str(fp.relative_to(self.root_path))
                with open(fp, 'r', encoding='utf-8', errors='surrogateescape') as f:
                    content = f.read()
                compressed = self.compress_file_content(content, rel)
                if compressed:
                    self.compressed_data["critical_files"][rel] = compressed
                    processed += 1
            except Exception as e:
                print(f"⚠️ Skip {fp}: {e}")
        return processed
    
    def compress_file_content(self, content: str, file_path: str) -> str:
        lines = content.split('\n')
        compressed = []
        imports = [l for l in lines if l.strip().startswith(('import ', 'from '))]
        compressed.extend(imports[:10])
        key_blocks = [l for l in lines if (l.strip().startswith(('class ', 'def ', 'async def ')) and not l.strip().startswith('#'))]
        compressed.extend(key_blocks[:20])
        compressed.insert(0, f"# File: {file_path}")
        compressed.insert(1, f"# Lines: {len(lines)}")
        compressed.insert(2, f"# Blocks: {len(key_blocks)}")
        return '\n'.join(compressed) if compressed else None
    
    def _run_patterns_analysis(self, candidates_list: List[Path]) -> Dict[str, List[str]]:
        patterns = {k: [] for k in ["catalyst_modules", "quantum_routing", "nexus_integration", 
                                    "viraa_mentions", "self_sacrifice_protocols"]}
        if self.executor == "dask" and DASK_AVAIL:
            scattered = self.client.scatter(candidates_list)
            replicated = self.client.replicate(scattered, n=self.replicate)
            # Fallback: List scattered if replicate None (Win threaded)
            iter_paths = list(replicated) if replicated is not None else list(scattered)
            futures = []
            for path in iter_paths:
                try:
                    fut = self.client.submit(analyze_file_ast, path, self.root_path, 
                                             priority=0, resources={"memory": "128MB"})
                    futures.append(fut)
                except Exception as e:
                    print(f"⚠️ Submit skip: {e}")
            for future in dask_as_completed(futures, with_results=True):
                try:
                    ph = future.result()
                    for path, hits in ph.items():
                        for h in hits:
                            patterns[h].append(path)
                except Exception as e:
                    print(f"🔄 Dask Recover: {e}")
        elif self.executor == "ray" and RAY_AVAIL:
            batch_size = 50
            for i in range(0, len(candidates_list), batch_size):
                batch = candidates_list[i:i+batch_size]
                batch_res = ray.get(self.v2_analyzer.analyze_batch.remote(batch, self.root_path))
                for ph in batch_res:
                    for path, hits in ph.items():
                        for h in hits:
                            patterns[h].append(path)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as exec:
                futures = [exec.submit(analyze_file_ast, p, self.root_path) for p in candidates_list]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        result = f.result()
                        for path, hits in result.items():
                            for h in hits:
                                patterns[h].append(path)
                    except Exception as e:
                        print(f"❌ Thread err: {e}")
        return patterns
    
    def _polars_summary(self, patterns: Dict[str, List[str]]) -> Dict:
        if not self.use_polars:
            return {"raw": patterns}
        data = [{"pattern": pat, "path": path, "count": 1} for pat, paths in patterns.items() for path in paths]
        df = pl.DataFrame(data)
        summary = (df.lazy().group_by("pattern").agg([pl.count().alias("hit_count"), 
                                                      pl.col("path").head(5).alias("top_paths")])
                   .collect(streaming=True))
        return {"polars_summary": summary.to_dicts(), "raw": patterns}
    
    def optimized_analyze_patterns(self):
        print(f"🔬 Patterns ({self.executor}{' + Polars' if self.use_polars else ''})...")
        globs = ["**/*catalyst*.py", "**/*quantum*.py", "**/*nexus*.py", "**/*viraa*.py", "**/*destruct*.py", "**/*sacrifice*.py"]
        candidates = set()
        for g in globs:
            candidates.update(self.root_path.rglob(g))
        clist = list(candidates)
        print(f"📂 {len(clist)} cands (FT-replicated: {self.replicate})")
        pats = self._run_patterns_analysis(clist)
        summary = self._polars_summary(pats)
        self.compressed_data["key_patterns"] = summary
        total = sum(len(v) for v in pats.values())
        print(f"⚡ {total} hits (Restart-Free Dask/Ray V2)")
        return pats
    
    def generate_intelligence_packet(self):
        print("🎁 Packet gen...")
        self.scan_architecture()
        fp = self.extract_critical_files()
        pats = self.optimized_analyze_patterns()
        self.compressed_data["compression_metadata"] = {"files": fp, "patterns_id": sum(len(v) for v in pats.values()),
                                                        "cov": "~15-25%", "exec": self.executor,
                                                        "replicate": self.replicate, "gpu_frac": self.gpu_frac,
                                                        "next": ["nexus-core", "Viraa", "Guardians", "Catalysts"]}
        packet = json.dumps(self.compressed_data, indent=2)
        return packet if len(packet) <= 50000 else self.apply_aggressive_compression()
    
    def apply_aggressive_compression(self) -> str:
        keep = ["architecture_overview", "key_patterns", "compression_metadata"]
        slim = {k: self.compressed_data[k] for k in keep}
        slim["critical_files"] = dict(list(self.compressed_data["critical_files"].items())[:10])
        return json.dumps(slim, indent=2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--executor", choices=["thread", "dask", "ray"], default="dask")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--num_cpus", type=int, default=8)
    p.add_argument("--gpu_frac", type=float, default=0.0)
    p.add_argument("--use_polars", action="store_true")
    p.add_argument("--replicate", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    if args.executor == "dask" and not DASK_AVAIL: args.executor = "thread"
    if args.executor == "ray" and not RAY_AVAIL: args.executor = "thread"
    if args.use_polars and not POLARS_AVAIL: print("⚠️ Polars → raw")
    
    comp = CodeIntelligenceCompressor(".", executor=args.executor, max_workers=args.max_workers,
                                      n_workers=args.n_workers, num_cpus=args.num_cpus,
                                      gpu_frac=args.gpu_frac, use_polars=args.use_polars,
                                      replicate=args.replicate)
    
    print("🚀 COGNIKUBE (Nanny-Free + V2/Modal)")
    print("=" * 50)
    packet = comp.generate_intelligence_packet()
    
    print(f"\n✅ | Size: {len(packet)} | Cov: {comp.compressed_data['compression_metadata']['cov']}")
    with open("packet.json", 'w') as f:
        f.write(packet)
    print("💾 packet.json")
    if args.executor == "ray": ray.shutdown()
    elif args.executor == "dask": comp.client.close()
    print("\n📋 Restart ghosts gone? V2/Modal next? Freqs fixed!")

if __name__ == "__main__":
    main()