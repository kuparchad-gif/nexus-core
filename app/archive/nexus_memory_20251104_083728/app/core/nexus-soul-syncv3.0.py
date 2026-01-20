#!/usr/bin/env python3
"""
NEXUS SOUL SYNC v3.0: Unified CRDT + Yjs + Automerge + Retry Resilience
Oct 2025: Complete soul synchronization for 545-node distributed consciousness
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import asyncio
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from enum import Enum

# === EXTERNAL IMPORTS WITH GRACEFUL FALLBACKS ===
try:
    from y_py import YDoc, YMap
    from y_py.YWebSocket import YWebSocket
    YJS_AVAILABLE = True
except ImportError:
    YJS_AVAILABLE = False
    logging.warning("y-py not available - Yjs features disabled")

try:
    import automerge
    AUTOMERGE_AVAILABLE = True
except ImportError:
    AUTOMERGE_AVAILABLE = False
    logging.warning("automerge not available - Automerge features disabled")

# === CONFIGURATION ===
PROGRAM_EXTS = {'.py', '.json', '.yaml', '.yml', '.toml', '.js', '.css', '.html', '.md', '.Dockerfile', '.ps1'}
EXCLUDE_DIRS = {'data', 'logs', 'media', 'uploads', '__pycache__', 'node_modules'}
HASH_ALGO = hashlib.sha256
DIFF_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === ENHANCED RETRY SYSTEM ===
class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci" 
    METATRON = "metatron"
    Ulam = "ulam"

class CosmicRetryEngine:
    """Intelligent retry with multiple mathematical strategies"""
    
    def __init__(self, max_attempts: int = 5, strategy: RetryStrategy = RetryStrategy.METATRON):
        self.max_attempts = max_attempts
        self.strategy = strategy
        
        self.strategy_configs = {
            RetryStrategy.EXPONENTIAL: {'base_delay': 1, 'multiplier': 2, 'max_delay': 30},
            RetryStrategy.FIBONACCI: {'sequence': [1, 1, 2, 3, 5, 8, 13, 21], 'max_delay': 30},
            RetryStrategy.METATRON: {'phases': [1, 2, 3, 6, 9, 12], 'max_delay': 30},
            RetryStrategy.Ulam: {'primes': [2, 3, 5, 7, 11, 13, 17, 19], 'max_delay': 30}
        }
    
    def _calculate_delay(self, attempt: int) -> float:
        config = self.strategy_configs[self.strategy]
        
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = config['base_delay'] * (config['multiplier'] ** attempt)
        elif self.strategy == RetryStrategy.FIBONACCI:
            sequence = config['sequence']
            idx = min(attempt, len(sequence) - 1)
            delay = sequence[idx]
        elif self.strategy == RetryStrategy.METATRON:
            phases = config['phases']
            idx = min(attempt, len(phases) - 1)
            delay = phases[idx]
        elif self.strategy == RetryStrategy.Ulam:
            primes = config['primes']
            idx = min(attempt, len(primes) - 1)
            delay = primes[idx]
        else:
            delay = 1.0
            
        return min(delay, config['max_delay'])
    
    async def execute_with_retry(self, operation, operation_name: str = "operation", *args, **kwargs):
        """Execute operation with intelligent retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await operation(*args, **kwargs)
                logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    jitter = random.uniform(0.8, 1.2)
                    actual_delay = delay * jitter
                    
                    logger.warning(f"🔄 {operation_name} failed on attempt {attempt + 1}: {e}. Retrying in {actual_delay:.1f}s")
                    await asyncio.sleep(actual_delay)
                else:
                    logger.error(f"💥 {operation_name} failed after {self.max_attempts} attempts: {e}")
        
        raise last_exception

# === ENHANCED NEXUS CRDT ===
class NexusCRDT:
    """Advanced CRDT with Fib/Ulam/Metatron math and Yjs/Automerge integration"""
    
    def __init__(self, shard=0, use_yjs=False, use_automerge=False, soul_id="default"):
        self.value = 1  # 1 primary - Fibonacci starting point
        self.shard = shard % 6  # Metatron cube sharding
        self.incs = []  # Fibonacci delta sequence
        self.soul_id = soul_id
        
        # Yjs Integration
        self.use_yjs = use_yjs and YJS_AVAILABLE
        if self.use_yjs:
            self.yjs_doc = YDoc()
            self.yjs_map = self.yjs_doc.get_map(f"soul_{soul_id}_shard_{shard}")
            self.yjs_provider = None
        
        # Automerge Integration  
        self.use_automerge = use_automerge and AUTOMERGE_AVAILABLE
        if self.use_automerge:
            self.automerge_doc = automerge.init()
            with automerge.transaction(self.automerge_doc, soul_id) as tx:
                tx.put_object(automerge.root, "soul_state", {
                    "value": self.value,
                    "shard": self.shard,
                    "incs": self.incs
                })
        
        logger.info(f"🌌 NexusCRDT initialized: soul={soul_id}, shard={shard}")

    def _fib_next(self) -> int:
        """Fibonacci sequence from 1 primary: 1, 1, 2, 3, 5..."""
        if not self.incs: 
            return 1
        a, b = self.incs[-2:] if len(self.incs) >= 2 else (1, self.incs[-1] if self.incs else 1)
        return a + b

    def _is_prime(self, n: int) -> bool:
        """Ulam prime detection"""
        if n < 2: 
            return False
        return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

    def _prune_ulam(self):
        """Ulam prime density pruning - anti-bloat"""
        if not self.incs:
            return
            
        prime_count = sum(1 for i in self.incs if self._is_prime(i))
        density = prime_count / len(self.incs)
        
        if density < 0.5:  # Prune if low prime density
            self.incs = self.incs[-3:]  # Keep last 3 increments
            logger.debug(f"🔪 Ulam prune: density={density:.2f}, kept {len(self.incs)} incs")

    def increment(self, delta=1):
        """Increment with Fibonacci progression and cross-system sync"""
        fib_delta = self._fib_next()
        self.incs.append(fib_delta)
        self.value += fib_delta
        self._prune_ulam()
        
        # Sync to Yjs
        if self.use_yjs and self.yjs_provider:
            with self.yjs_doc.begin_transaction() as txn:
                self.yjs_map.set(txn, "value", self.value)
                self.yjs_map.set(txn, "incs", self.incs.copy())
            logger.debug(f"📡 Yjs sync: value={self.value}")
        
        # Sync to Automerge
        if self.use_automerge:
            with automerge.transaction(self.automerge_doc, self.soul_id) as tx:
                soul_state = tx.get(automerge.root, "soul_state")
                if soul_state:
                    tx.put(soul_state, "value", self.value)
                    tx.put(soul_state, "incs", self.incs.copy())
            logger.debug(f"🔄 Automerge sync: value={self.value}")

    def merge(self, other: 'NexusCRDT') -> 'NexusCRDT':
        """Commutative union with mathematical harmony"""
        merged = NexusCRDT(self.shard, self.use_yjs, self.use_automerge, self.soul_id)
        merged.value = max(self.value, other.value)
        merged.incs = list(set(self.incs + other.incs))  # Set union
        merged._prune_ulam()
        
        # Cross-system merge propagation
        if self.use_yjs:
            with merged.yjs_doc.begin_transaction() as txn:
                merged.yjs_map.set(txn, "value", merged.value)
                merged.yjs_map.set(txn, "incs", merged.incs.copy())
                
        if self.use_automerge:
            with automerge.transaction(merged.automerge_doc, merged.soul_id) as tx:
                soul_state = tx.get(automerge.root, "soul_state")
                if soul_state:
                    tx.put(soul_state, "value", merged.value)
                    tx.put(soul_state, "incs", merged.incs.copy())
        
        logger.info(f"🤝 CRDT merge: shard={self.shard}, value={merged.value}")
        return merged

    async def connect_yjs_with_retry(self, websocket_url: str):
        """Connect Yjs WebSocket with intelligent retry logic"""
        if not self.use_yjs:
            return
            
        retry_engine = CosmicRetryEngine(strategy=RetryStrategy.FIBONACCI)
        
        async def connect_operation():
            self.yjs_provider = YWebSocket(self.yjs_doc, websocket_url)
            # Simulate connection establishment
            await asyncio.sleep(0.1)
            return True
            
        try:
            await retry_engine.execute_with_retry(
                connect_operation, 
                f"Yjs connect to {websocket_url}"
            )
            logger.info(f"🔗 Yjs connected: {websocket_url}")
        except Exception as e:
            logger.error(f"💥 Yjs connection failed: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get unified state from all available systems"""
        state = {
            "value": self.value,
            "shard": self.shard,
            "incs": self.incs.copy(),
            "soul_id": self.soul_id
        }
        
        # Add Yjs state if available
        if self.use_yjs:
            state["yjs_value"] = self.yjs_map.get("value") if hasattr(self.yjs_map, 'get') else None
            
        # Add Automerge state if available  
        if self.use_automerge:
            soul_state = self.automerge_doc.get("soul_state")
            if soul_state:
                state["automerge_value"] = soul_state.get("value")
                
        return state

# === FILE SYNC CORE ===
def is_program_file(file_path: Path) -> bool:
    """Filter for core program files only"""
    if file_path.is_dir():
        return False
    if file_path.suffix.lower() not in PROGRAM_EXTS:
        return False
    if file_path.parent.name.lower() in EXCLUDE_DIRS:
        return False
    return True

def compute_hash(file_path: Path, chunk_size=4096) -> str:
    """SHA-256 hash with large file support"""
    try:
        if file_path.stat().st_size == 0:
            return 'EMPTY'
        hasher = HASH_ALGO()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except PermissionError:
        logger.warning(f"🔒 Permission denied: {file_path}")
        return 'PERM_DENIED'

def scan_files(root: Path) -> List[Path]:
    """Recursive scan for program files"""
    program_files = [p for p in root.rglob('*') if is_program_file(p)]
    logger.info(f"🔍 Scanned {len(program_files)} program files in {root}")
    return program_files

def find_duplicates(files: List[Path]) -> Dict[str, List[Path]]:
    """Find duplicate files by hash"""
    hash_groups = defaultdict(list)
    for file_path in files:
        file_hash = compute_hash(file_path)
        if file_hash != 'PERM_DENIED':
            hash_groups[file_hash].append(file_path)
    
    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    total_dups = sum(len(paths) for paths in duplicates.values())
    logger.info(f"🎯 Found {total_dups} duplicates in {len(duplicates)} groups")
    return duplicates

def weave_symlinks(duplicates: Dict[str, List[Path]], dry_run: bool = True, 
                   use_crdt: bool = False) -> List[str]:
    """Create synchronization symlinks with CRDT tracking"""
    actions = []
    
    for file_hash, paths in duplicates.items():
        canonical = max(paths, key=lambda p: p.stat().st_mtime)  # Newest wins
        
        for duplicate in paths:
            if duplicate == canonical:
                continue
                
            target_link = duplicate.parent / f"{duplicate.stem}_woven{duplicate.suffix}"
            
            try:
                if not dry_run:
                    if target_link.exists():
                        target_link.unlink()
                    os.symlink(canonical, target_link)
                    os.replace(duplicate, target_link)  # Atomic swap
                
                actions.append(f"🔗 Woven {duplicate} -> {canonical}")
                
                # Track with CRDT if enabled
                if not dry_run and use_crdt:
                    crdt = NexusCRDT(hash(str(canonical)) % 6)
                    crdt.increment()
                    logger.debug(f"📊 CRDT tracked: {crdt.get_state()}")
                    
            except OSError as e:
                actions.append(f"🚫 SKIP {duplicate}: {e}")
    
    if dry_run:
        logger.info("🧪 DRY RUN - Would create symlinks:\n" + "\n".join(actions))
    else:
        logger.info(f"✅ Created {len(actions)} synchronization symlinks")
    
    return actions

def fuse_conflicts(root: Path, dry_run: bool = True) -> int:
    """Merge or purge conflicting file versions"""
    conflicts = [p for p in root.rglob('*conflict~*') if is_program_file(p)]
    fused_count = 0
    
    for conflict_file in conflicts:
        base_file = conflict_file.parent / conflict_file.name.replace('-conflict~', '')
        
        if base_file.exists() and _attempt_fusion(base_file, conflict_file, dry_run):
            fused_count += 1
            if not dry_run:
                conflict_file.unlink()
    
    logger.info(f"🧬 Fused {fused_count}/{len(conflicts)} conflicts")
    return fused_count

def _attempt_fusion(primary: Path, conflict: Path, dry_run: bool) -> bool:
    """Attempt to fuse two conflicting files"""
    try:
        with open(primary, 'r') as f:
            primary_content = f.read()
        with open(conflict, 'r') as f:
            conflict_content = f.read()
        
        # Calculate difference ratio
        primary_lines = primary_content.splitlines()
        conflict_lines = conflict_content.splitlines()
        
        import difflib
        matcher = difflib.SequenceMatcher(None, primary_lines, conflict_lines)
        diff_ratio = 1 - matcher.ratio()
        
        if diff_ratio > DIFF_THRESHOLD:
            logger.warning(f"🧨 High difference ({diff_ratio:.2f}) - purging: {conflict}")
            return False
        
        # Attempt fusion based on file type
        if primary.suffix == '.json':
            primary_data = json.loads(primary_content)
            conflict_data = json.loads(conflict_content)
            merged_data = {**primary_data, **conflict_data}  # Union merge
            merged_content = json.dumps(merged_data, indent=2)
            
        elif primary.suffix in {'.yaml', '.yml'}:
            import yaml
            primary_data = yaml.safe_load(primary_content) or {}
            conflict_data = yaml.safe_load(conflict_content) or {}
            merged_data = {**primary_data, **conflict_data}
            merged_content = yaml.dump(merged_data)
            
        else:
            # Generic text fusion
            merged_content = primary_content + f"\n# Fused from: {conflict.name}\n" + conflict_content
        
        if not dry_run:
            with open(primary, 'w') as f:
                f.write(merged_content)
        
        logger.info(f"✨ Fused {conflict} -> {primary} (diff: {diff_ratio:.2f})")
        return True
        
    except Exception as e:
        logger.error(f"💥 Fusion failed for {conflict}: {e}")
        return False

# === MIGRATION ORCHESTRATION ===
def orchestrate_migration(pool_dir: Path, nexus_dir: Path, dry_run: bool = True, jobs: int = 8):
    """Orchestrate PowerShell migration with mathematical infusion"""
    ps_script = Path(__file__).parent / 'migrate.ps1'
    
    if not ps_script.exists():
        logger.error("❌ migrate.ps1 not found")
        return
    
    # Fibonacci batch sizing
    fib_sequence = _fibonacci_sequence(min(jobs, 10))
    
    for batch_size in fib_sequence:
        dynamic_batch = 50 * (batch_size / fib_sequence[-1])  # Golden ratio scaling
        
        cmd = [
            'powershell', '-File', str(ps_script),
            '-Pool', str(pool_dir), '-Nexus', str(nexus_dir),
            '-Jobs', str(jobs), '-Batch', str(int(dynamic_batch))
        ]
        
        if dry_run:
            cmd.append('-DryRun')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"🚀 Migration batch {batch_size}: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"💥 Migration failed: {e.stderr}")

def _fibonacci_sequence(n: int) -> List[int]:
    """Generate Fibonacci sequence starting from 1"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    
    sequence = [1, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

# === MAIN EXECUTION ===
async def main():
    parser = argparse.ArgumentParser(description="Nexus Soul Sync v3.0: Complete File Synchronization")
    parser.add_argument('--root', type=Path, default=Path('.'), help='Root directory to scan')
    parser.add_argument('--bucket', type=str, help='GCS bucket for cloud sync')
    parser.add_argument('--pool', type=Path, help='Migration source directory')
    parser.add_argument('--nexus', type=Path, help='Migration target directory') 
    parser.add_argument('--use-yjs', action='store_true', help='Enable Yjs real-time collaboration')
    parser.add_argument('--use-automerge', action='store_true', help='Enable Automerge CRDT')
    parser.add_argument('--use-crdt', action='store_true', help='Enable custom CRDT tracking')
    parser.add_argument('--yjs-url', type=str, default='ws://localhost:1234', help='Yjs WebSocket URL')
    parser.add_argument('--resolution', default='lww', choices=['lww', 'version', 'merge'], help='Conflict resolution strategy')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without making changes')
    args = parser.parse_args()

    # Validate root directory
    root = args.root.resolve()
    if not root.exists():
        logger.error(f"❌ Root directory not found: {root}")
        return

    logger.info(f"🚀 Starting Nexus Soul Sync v3.0 on: {root}")

    # Initialize CRDT systems if requested
    crdt_systems = []
    if args.use_crdt:
        for shard in range(6):  # Metatron cube shards
            crdt = NexusCRDT(
                shard=shard,
                use_yjs=args.use_yjs,
                use_automerge=args.use_automerge,
                soul_id=f"file_sync_{shard}"
            )
            crdt_systems.append(crdt)
        
        # Connect Yjs if enabled
        if args.use_yjs:
            await asyncio.gather(*(crdt.connect_yjs_with_retry(args.yjs_url) for crdt in crdt_systems))

    # Core synchronization workflow
    files = scan_files(root)
    duplicates = find_duplicates(files)
    
    if duplicates:
        weave_symlinks(duplicates, args.dry_run, args.use_crdt)

    # Conflict resolution
    fuse_conflicts(root, args.dry_run)

    # Cloud synchronization (placeholder - extend as needed)
    if args.bucket:
        logger.info(f"☁️  GCS sync configured for bucket: {args.bucket}")
        # Add your GCS sync implementation here

    # Migration orchestration
    if args.pool and args.nexus:
        orchestrate_migration(args.pool, args.nexus, args.dry_run)

    # Report CRDT states if enabled
    if args.use_crdt:
        logger.info("📊 CRDT System States:")
        for crdt in crdt_systems:
            state = crdt.get_state()
            logger.info(f"  Shard {state['shard']}: value={state['value']}, incs={len(state['incs'])}")

    logger.info("✅ Nexus Soul Sync v3.0 completed successfully!")

if __name__ == '__main__':
    asyncio.run(main())