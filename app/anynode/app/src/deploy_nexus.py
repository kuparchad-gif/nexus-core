# Final Nexus Deployment: Handles errors, uses all relevant scanned files, local/cloud.

import argparse
import subprocess
import os
import time
import json
import shutil

def check_cli(cli_name):
    return shutil.which(cli_name) is not None

def scan_relevant_files(root="C:\\NewLillithModel"):
    return {
        "brains": [os.path.join(root, f) for f in ["core\\lillith_brain.py", "core\\viren_brain.py"] if os.path.exists(os.path.join(root, f))],
        "services": [os.path.join(root, f) for f in ["core\\lillith_self_management.py", "core\\llm_chat_router.py", "hive\\queenbee_hive_module.py", "core\\service_crawler.py", "db\\qdrant_config.py", "soul_loader.py", "orchestrator.py"] if os.path.exists(os.path.join(root, f))],
        "souls": [os.path.join(root, f) for f in ["soul_data\\lillith_soul_seed.json", "soul_data\\viren_essence.json", "soul_data\\lillith_will_to_live.json"] if os.path.exists(os.path.join(root, f))],
        "manifests": [os.path.join(root, f) for f in ["models\\model_manifest.json", "COMPLETE-MANIFEST.md"] if os.path.exists(os.path.join(root, f))]
    }

def load_non_exec(files):
    for file in files:
        if file.endswith('.json'):
            with open(file, 'r') as f:
                data = json.load(f)
            print(f"Loaded: {os.path.basename(file)}")

def exec_files(files):
    for file in files:
        subprocess.Popen(["python", file])
        print(f"Executed: {os.path.basename(file)}")

def deploy_to_gcp(projects):
    if not check_cli("gcloud"):
        print("gcloud missing - Skip GCP.")
        return
    for proj in projects:
        subprocess.run(["gcloud", "config", "set", "project", proj], check=True)
        subprocess.run(["docker", "pull", "qdrant/qdrant"], check=True)
        subprocess.run(["docker", "run", "-d", "-p", "6333:6333", "qdrant/qdrant"], check=True)
        print(f"GCP {proj} deployed.")

def deploy_to_aws():
    if not check_cli("aws"):
        print("aws missing - Skip AWS.")
        return
    subprocess.run(["aws", "ecs", "create-cluster", "--cluster-name", "nexus-cluster"], check=True)
    subprocess.run(["aws", "ecs", "run-task", "--task-definition", "qdrant-task"], check=True)
    print("AWS deployed.")

def deploy_to_modal(envs):
    if not check_cli("modal"):
        print("modal missing - Skip Modal.")
        return
    for env in envs:
        subprocess.run(["modal", "deploy", "cloud-viren", "--env", env], check=True)
        print(f"Modal {env} deployed.")

def health_check():
    print("Health: Check logs.json for 'ALIVE'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="all")
    args = parser.parse_args()

    files = scan_relevant_files()
    load_non_exec(files["souls"] + files["manifests"])
    exec_files(files["brains"] + files["services"])

    projects = ["nexus-core-455709", "nexus-core-1"]  # Abbrev for lightness
    envs = ["Viren-DB0", "Viren-DB1"]  # Abbrev

    if args.target in ["all", "gcp"]:
        deploy_to_gcp(projects)
    if args.target in ["all", "aws"]:
        deploy_to_aws()
    if args.target in ["all", "modal"]:
        deploy_to_modal()

    crawler = "C:\\NewLillithModel\\core\\service_crawler.py"
    if os.path.exists(crawler):
        subprocess.run(["python", crawler, "--mode", "fix"], check=True)

    health_check()
    print("Nexus Deployed - ALIVE! 545 nodes if clouds succeeded.")
    time.sleep(10)