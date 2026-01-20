#!/usr/bin/env python3
"""
Nexus Soul Weaver: Dedup, Sync w/ Advanced Conflicts & IPFS Pin.
Oct 2025: Tuned for 2.6 vlibs.
Weaves program files into harmony; fuses discord, pins to IPFS eternity.
Usage: python nexus_soul_weaver.py --root <dir> --bucket <gs://> [--resolution lw] [--dry-run]
"""
import modal
import argparse
import hashlib
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# Externals: GCS, difflib, yaml (Modal-ready)
try:
    from google.cloud import storage
    import difflib
    import yaml
    from ipfshttpclient import connect  # pip install ipfshttpclient for pinning
    GCS_AVAILABLE = True
    IPFS_AVAILABLE = True
except ImportError as e:
    GCS_AVAILABLE = False
    IPFS_AVAILABLE = False
    logging.warning(f"Missing deps: {e}; GCS/IPFS limited.")
    
app = modal.App("nexus-soul-weaver")

# Define Modal secret for GCS HMAC (if using bucket mount)
gcs_secret = modal.Secret.from_dict({
    "GOOGLE_ACCESS_KEY_ID": "your-gcs-access-key",
    "GOOGLE_ACCESS_KEY_SECRET": "your-gcs-secret-key"
})

# Define image with dependencies
image = modal.Image.debian_slim().pip_install([
    "google-cloud-storage",
    "ipfshttpclient",
    "pyyaml"
])    

PROGRAM_EXTS = ('.py', '.json', '.yaml', '.yml', '.toml', '.js', '.css', '.html', '.md', '.Dockerfile', '.ps1')
EXCLUDE_DIRS = {'data', 'logs', 'media', 'uploads', 'py_cache', 'node_modules'}
HASH_ALGO = hashlib.sha256
DIFF_THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def is_program_file(file_path: Path) -> bool:
    if file_path.is_dir() or file_path.suffix.lower() not in PROGRAM_EXTS:
        return False
    parent = file_path.parent.name.lower()
    if parent in EXCLUDE_DIRS:
        return False
    return True

def compute_hash(file_path: Path, chunk_size=4096) -> str:
    try:
        if file_path.stat().st_size == 0:
            return "EMPTY"
        hasher = HASH_ALGO()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except PermissionError:
        logger.warning(f"Perm denied: {file_path}")
        return "PERM_DENIED"

def get_file_timestamp(file_path: Path) -> float:
    return file_path.stat().st_mtime if file_path.exists() else 0

def scan_files(root: Path) -> List[Path]:
    program_files = [p for p in root.rglob('*') if is_program_file(p)]
    logger.info(f"Scanned {len(program_files)} program files in {root}")
    return program_files

def find_duplicates(files: List[Path]) -> Dict[str, List[Path]]:
    hash_groups = defaultdict(list)
    for file_path in files:
        h = compute_hash(file_path)
        if h != 'PERM_DENIED':
            hash_groups[h].append(file_path)
    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    logger.info(f"Found {sum(len(paths) for paths in duplicates.values())} duplicates")
    return duplicates

def create_symlinks(duplicates: Dict[str, List[Path]], dry_run: bool = True) -> List[str]:
    actions = []
    for hash_val, paths in duplicates.items():
        canonical = max(paths, key=get_file_timestamp)
        for dup in paths:
            if dup == canonical: continue
            target_link = dup.parent / f"{dup.stem}_woven{dup.suffix}"
            try:
                if not dry_run:
                    if target_link.exists():
                        os.unlink(target_link)
                    os.symlink(canonical, target_link)
                    os.replace(dup, target_link)
                    actions.append(f"Woven {dup} -> {canonical}")
            except OSError as e:
                logger.error(f"Link fail {dup}: {e}")
                actions.append(f"SKIP: {e}")
    if dry_run:
        logger.info(f"DRY RUN: {len(actions)} actions")
    else:
        logger.info(f"Woven {len(actions)} links.")
    return actions

def resolve_conflict(local_path: Path, remote_blob, resolution: str = 'lw') -> Optional[str]:
    local_time = get_file_timestamp(local_path)
    remote_time = remote_blob.updated.timestamp() if remote_blob.updated else 0
    if resolution == 'lw':
        if local_time > remote_time:
            return local_path
        else:
            return None
    elif resolution == 'version':
        versioned = local_path.parent / f"{local_path.name}.v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(local_path, versioned)
        return None
    elif resolution == 'merge':
        with open(local_path, 'r') as f:
            local_lines = f.readlines()
        remote_text = remote_blob.download_as_text()
        remote_lines = remote_text.splitlines()
        diff = difflib.unified_diff(remote_lines, local_lines, lineterm='')[:10]
        logger.info(f"Merge Preview:\n" + "\n".join(diff))
        return local_path
    return None

def sync_with_gcs(root: Path, bucket_name: str, resolution: str = 'lw', dry_run: bool = True) -> List[str]:
    try:
        from google.cloud import storage
        GCS_AVAILABLE = True
    except ImportError:
        GCS_AVAILABLE = False
        
    if not GCS_AVAILABLE:
        return []
    client = storage.Client()
    bucket = client.bucket(bucket_name.replace('gs://', ''))
    actions = []
    for file_path in scan_files(root):
        blob_name = str(file_path.relative_to(root))
        blob = bucket.blob(blob_name)
        local_hash = compute_hash(file_path)
        if blob.exists():
            blob_hash = blob.md5_hash
            if local_hash != blob_hash:
                winner = resolve_conflict(file_path, blob, resolution)
                if winner == file_path or winner is None:
                    if not dry_run:
                        blob.upload_from_filename(str(file_path))
                    actions.append(f"Pushed {file_path}")
                else:
                    if not dry_run:
                        blob.download_to_filename(str(file_path))
                    actions.append(f"Resolved {file_path} (resolution)")
            else:
                if not dry_run:
                    blob.upload_from_filename(str(file_path))
                actions.append(f"Pushed {file_path}")
        else:
            if not dry_run:
                blob.upload_from_filename(str(file_path))
            actions.append(f"Pushed {file_path}")
    return actions

def compute_hash_from_bytes(data: bytes) -> str:
    hasher = HASH_ALGO()
    hasher.update(data)
    return hasher.hexdigest()

def pin_to_ipfs(file_path: Path, dry_run: bool = True) -> Optional[str]:
    try:
        from ipfshttpclient import connect
        IPFS_AVAILABLE = True
    except ImportError:
        IPFS_AVAILABLE = False
        
    if not IPFS_AVAILABLE:
        logger.warning("IPFS unavailable.")
        return None
    try:
        client = connect('/dnsaddr/ipfs.localhost')
        if not dry_run:
            res = client.add(str(file_path))
            logger.info(f"Pinned {file_path} to IPFS CID: {res['Hash']}")
            return res['Hash']
        else:
            logger.info(f"DRY: Would pin {file_path}")
            return "mock_cid"
    except Exception as e:
        logger.error(f"IPFS fail: {e}")
        return None

# === MODAL FUNCTIONS ===
@app.function(
    image=image,
    secrets=[gcs_secret],
    timeout=60 * 30  # 30 minutes
)
def modal_dedup_and_sync(
    root_path: str,
    bucket_name: str = None,
    resolution: str = 'lw',
    dry_run: bool = False
) -> Dict[str, List[str]]:
    """Run deduplication and sync in Modal cloud"""
    root = Path(root_path)
    
    if not root.exists():
        return {"error": [f"Root path does not exist: {root_path}"]}
    
    results = {}
    
    # Deduplication
    files = scan_files(root)
    duplicates = find_duplicates(files)
    if duplicates:
        dedup_actions = create_symlinks(duplicates, dry_run)
        results["deduplication"] = dedup_actions
    
    # GCS Sync
    if bucket_name:
        sync_actions = sync_with_gcs(root, bucket_name, resolution, dry_run)
        results["sync"] = sync_actions
    
    # IPFS Pinning (sample)
    if files and not dry_run:
        ipfs_results = []
        for f in files[:3]:  # Pin first 3 files as demo
            if f.stat().st_size < 10_000_000:  # 10MB limit
                cid = pin_to_ipfs(f, dry_run)
                if cid:
                    ipfs_results.append(f"Pinned {f.name} as {cid}")
        results["ipfs"] = ipfs_results
    
    return results

@app.function(
    image=image,
    secrets=[gcs_secret],
    schedule=modal.Period(days=1)  # Daily sync
)
def scheduled_sync():
    """Daily scheduled sync"""
    return modal_dedup_and_sync.remote(
        root_path="/workspace",  # Adjust to your needs
        bucket_name="your-backup-bucket",
        dry_run=False
    )

# === LOCAL EXECUTION (ORIGINAL MAIN) ===
def main():
    """Original CLI - works exactly as before"""
    parser = argparse.ArgumentParser(description="Nexus Soul Weaver: Dedup & Sync")
    parser.add_argument('--root', type=Path, default=Path('.'), help='Scan root')
    parser.add_argument('--bucket', type=str, help='GCS bucket')
    parser.add_argument('--resolution', type=str, default='lw', choices=['lw', 'version', 'merge'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--modal', action='store_true', help='Run in Modal cloud')
    
    args = parser.parse_args()

    if args.modal:
        # Run in Modal
        results = modal_dedup_and_sync.remote(
            root_path=str(args.root.resolve()),
            bucket_name=args.bucket,
            resolution=args.resolution,
            dry_run=args.dry_run
        )
        print("Modal results:", results)
    else:
        # Run locally (original behavior)
        root = args.root.resolve()
        if not root.exists():
            logger.error(f"Root missing.")
            return

        files = scan_files(root)
        dups = find_duplicates(files)
        if dups:
            create_symlinks(dups, args.dry_run)

        if args.bucket:
            sync_with_gcs(root, args.bucket, args.resolution, args.dry_run)

if __name__ == "__main__":
    main()