import argparse
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping patterns: file name/pattern -> target relative path
PATH_MAPPING: Dict[str, str] = {
    'env.modes.json': 'Config/env.modes.json',
    'ship_manifest.json': 'Config/ship_manifest.json',
    'sovereignty.yaml': 'Config/policy/sovereignty.yaml',
    'council_rules.yaml': 'Config/policy/council_rules.yaml',
    'theme.tokens.json': 'Config/ui/theme.tokens.json',
    'layout.ratio.json': 'Config/ui/layout.ratio.json',
    'awaken_eden.py': 'Systems/nexus_runtime/heart/awaken_eden.py',
    'lillith_bootstrap.py': 'Systems/nexus_runtime/heart/lillith_bootstrap.py',
    'ignite.sh': 'Systems/nexus_runtime/ignite/ignite.sh',
    'ignite_eden.py': 'Systems/nexus_runtime/ignite/ignite_eden.py',
    'ignite_ship.py': 'Systems/nexus_runtime/ignite/ignite_ship.py',
    'env_detect.py': 'Systems/nexus_runtime/common_boot/env_detect.py',
    'health.py': 'Systems/nexus_runtime/common_boot/health.py',
    'orc': 'Systems/engine/orc/',  # Prefix for orc subfiles
    'mcp_tools': 'Systems/engine/orc/mcp_tools/',
    'ws': 'Systems/engine/orc/ws/',
    'planner': 'Systems/engine/planner/',
    'pulse': 'Systems/engine/pulse/',
    'memory': 'Systems/engine/memory/',
    'text': 'Systems/engine/text/',
    'tone': 'Systems/engine/tone/',
    'mythrunner': 'Systems/engine/mythrunner/',
    'ego': 'Systems/engine/mythrunner/ego/',
    'dream': 'Systems/engine/mythrunner/dream/',
    'guardian': 'Systems/engine/guardian/',
    'trinity_gate': 'Systems/engine/security/trinity_gate/',
    'towers': 'Systems/engine/towers/',
    'trinity_tower': 'Systems/engine/towers/trinity_tower/',
    'autonomic': 'Systems/engine/autonomic/',
    'viren': 'Systems/engine/autonomic/viren/',
    'observability': 'Systems/engine/observability/',
    'loki': 'Systems/engine/observability/loki/',
    'lillith': 'Systems/services/lillith/',
    'archiver': 'Systems/services/archiver/',
    'metatron': 'Systems/services/metatron/',
    'proto': 'Systems/adapters/proto/',
    'bridges': 'Systems/adapters/bridges/',
    'clients': 'Systems/adapters/clients/',
    'protocols': 'Systems/common/protocols/',
    'events': 'Systems/common/events/',
    'utils': 'Systems/common/utils/',
    'llm_core': 'Utilities/llm_core/',
    'llm_service.py': 'Utilities/llm_core/llm_service.py',
    'scripts': 'Utilities/scripts/',
    'monitoring': 'Utilities/monitoring/',
    'eden-portal': 'Web/eden-portal/',
    'lillith-console': 'Web/lillith-console/',
    'index': 'memory/index/',
    'cards': 'memory/cards/',
    'snapshots': 'memory/snapshots/',
    'qdrant': 'storage/qdrant/',
    'redis': 'storage/redis/',
    'logs': 'storage/logs/',
    'tmp': 'storage/tmp/',
    'compose.yaml': 'docker/compose.yaml',
    'Dockerfile.meta': 'docker/Dockerfile.meta',
    'base': 'k8s/base/',
    'overlays': 'k8s/overlays/',
    'e2e': 'tests/e2e/',
    'ADRs': 'docs/ADRs/',
    'runbooks': 'docs/runbooks/',
    '.env.example': '.env.example',
    'requirements.txt': 'requirements.txt',
    'Dockerfile': 'Dockerfile',
    'README.md': 'README.md',
    'ignition': 'ignition',
    # Add more specific mappings from repo files, e.g.
    'generate_trust_phases.py': 'Utilities/scripts/generate_trust_phases.py',
    'ws_spine.py': 'Systems/engine/orc/ws/ws_spine.py',
    'viren_ai.py': 'Systems/engine/autonomic/viren/viren_ai.py',
    # ... extend as needed for all known files
}

def determine_target(file_path: Path) -> Optional[str]:
    """Map file to target relative path based on name/parent."""
    name = file_path.name.lower()
    parent = file_path.parent.name.lower()
    for key, target in PATH_MAPPING.items():
        if key in name or key in parent:
            if target.endswith('/'):
                return target + name
            return target
    logger.warning(f"No mapping for {name}; defaulting to root/{name}")
    return name  # Root default

def create_structure(new_dir: Path):
    """Create all required subdirs upfront."""
    for target in set(PATH_MAPPING.values()):
        if target.endswith('/'):
            (new_dir / target).mkdir(parents=True, exist_ok=True)

def handle_duplicate(src: Path, dst: Path, force: bool) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)
        logger.info(f"Copied: {src} -> {dst}")
        return

    src_mtime = src.stat().st_mtime
    dst_mtime = dst.stat().st_mtime
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size

    if src_mtime > dst_mtime or (src_mtime == dst_mtime and src_size > dst_size):
        if force:
            bak = dst.with_suffix(dst.suffix + f".bak_{int(src_mtime)}")
            shutil.copy2(dst, bak)
            logger.info(f"Backed up: {dst} -> {bak}")
        shutil.copy2(src, dst)
        logger.info(f"Overwrote (newer/larger): {src} -> {dst}")
    else:
        logger.info(f"Skipped (existing newer/larger): {src}")

def organize_files(old_dir: str, new_dir: str, force: bool = False, dry_run: bool = False) -> None:
    old_path = Path(old_dir).resolve()
    new_path = Path(new_dir).resolve()

    if not old_path.is_dir():
        raise ValueError(f"Old directory not found: {old_path}")

    create_structure(new_path)

    for root, _, files in os.walk(old_path):
        for file in files:
            src = Path(root) / file
            if src.is_file():
                rel_target = determine_target(src)
                if rel_target:
                    dst = new_path / rel_target
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dry_run:
                        logger.info(f"Dry-run: Would copy {src} -> {dst}")
                    else:
                        try:
                            handle_duplicate(src, dst, force)
                        except Exception as e:
                            logger.error(f"Error processing {src}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Migrate and organize files to new structure.")
    parser.add_argument('--old', required=True, help="Path to old repository directory.")
    parser.add_argument('--new', required=True, help="Path to new repository directory.")
    parser.add_argument('--force', action='store_true', help="Force overwrite with backups.")
    parser.add_argument('--dry-run', action='store_true', help="Simulate without changes.")
    args = parser.parse_args()

    try:
        organize_files(args.old, args.new, args.force, args.dry_run)
        logger.info("Migration complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()