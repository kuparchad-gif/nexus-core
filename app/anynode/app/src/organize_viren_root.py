
import os
import shutil

ROOT = "."

moves = {
    # Move to boot
    "viren_chat.py": "boot/",
    "viren_platinum.py": "boot/",
    "viren_platinum_integration.py": "boot/",
    "viren_platinum_integration_complete.py": "boot/",
    "viren_platinum_interface.py": "boot/",
    "viren_platinum_mcp.py": "boot/",
    "bootstrap_environment.py": "boot/init/",
    "bootstrap_lillith.py": "boot/init/",
    "bootstrap_lillith_soul.py": "boot/init/",
    "bootstrap_viren.py": "boot/init/",

    # Move to Services
    "viren_voice_interface.py": "Services/voice/",
    "viren_stt.py": "Services/voice/",
    "viren_tts.py": "Services/voice/",
    "viren_document_suite.py": "Services/documents/",

    # Move to Utilities
    "check_nova_refs.py": "Utilities/",
    "migration_helper.py": "Utilities/",
    "security_hardening.py": "Utilities/security/",

    # Move to Scripts
    "install_dependencies.bat": "scripts/",
    "launch_engineers.bat": "scripts/launch/",

    # Move to Sandbox
    "conversation_router_visualizer.py": "sandbox/",

    # Move to Archive
    "viren_bootstrap.log": "archive/logs/",
    "SWARM_README.md": "archive/docs/",
    "LICENSE.txt": "archive/docs/",
    "cleanup_report.md": "archive/docs/",
    "mcp_log.txt": "archive/logs/",
    ".lmstudio-home-pointer": "archive/configs/",
    "template_engineer_memory.json": "archive/configs/",
    "rename_nova_references.py": "archive/legacy_scripts/",
    "simple_viren_chat.py": "archive/legacy_scripts/",
    "lillith_anatomy.md": "archive/docs/",
    "environment_context.json": "Config/",
    "models_to_load.txt": "Config/",
    "README_PLATINUM.md": "archive/docs/",

    # Move GitHub/Auth tools
    "github_client.py": "Services/github/",
    "github_interface.py": "Services/github/",
    "auth_manager.py": "Services/auth/",
    "cloud_connection.py": "Services/auth/",

    # Move Grays + Model Management
    "grays_anatomy.py": "Systems/engine/",
    "model_manager.py": "Systems/engine/",
    "model_service.py": "Systems/engine/",

    # Move Kubernetes YAMLs
    "lillith-prime-deployment.yaml": "Config/kubernetes/",
    "lillith-prime-service.yaml": "Config/kubernetes/",
}

for filename, target_dir in moves.items():
    src = os.path.join(ROOT, filename)
    dst = os.path.join(ROOT, target_dir, filename)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        print(f"Moving {filename} -> {target_dir}")
        shutil.move(src, dst)
    else:
        print(f"Skipped (not found): {filename}")
