import os
import sys
import argparse
import subprocess
import asyncio
import yaml
import logging
import requests

logging.basicConfig(filename='logs/boot.log', level=logging.INFO)
logger = logging.getLogger()

def log_message(message, is_verbose=False, verbose=False):
    logger.info(message)
    if verbose:
        print(message)

async def init_hermes():
    try:
        response = requests.get('http://localhost:11435/health')
        if response.status_code != 200:
            subprocess.run(['python', 'src/utils/hermes_start.py'])
        log_message("Hermes OS initialized", is_verbose=True)
    except Exception as e:
        log_message(f"Hermes init failed: {e}")

def load_configs(root):
    configs = {}
    for dirpath, _, files in os.walk(os.path.join(root, 'config')):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                with open(os.path.join(dirpath, file), 'r') as f:
                    configs[file] = yaml.safe_load(f)
            elif file.endswith('.json'):
                with open(os.path.join(dirpath, file), 'r') as f:
                    configs[file] = json.load(f)
            elif file.endswith('.env'):
                with open(os.path.join(dirpath, file), 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value
    log_message("Configs loaded")
    return configs

def prep_data(root):
    proc_dir = os.path.join(root, 'src/data_processing')
    for dirpath, _, files in os.walk(proc_dir):
        for file in files:
            if file.endswith('.py'):
                subprocess.run(['python', os.path.join(dirpath, file)])
            elif file.endswith('.js'):
                subprocess.run(['node', os.path.join(dirpath, file)])
    log_message("Data prepped")

def exec_notebooks(root):
    nb_dir = os.path.join(root, 'notebooks')
    for dirpath, _, files in os.walk(nb_dir):
        for file in files:
            if file.endswith('.ipynb'):
                subprocess.run(['jupyter', 'nbconvert', '--execute', '--to', 'notebook', '--inplace', os.path.join(dirpath, file)])
    log_message("Notebooks executed")

def exec_models_utils(root):
    for sub in ['models', 'utils']:
        sub_dir = os.path.join(root, f'src/{sub}')
        for dirpath, _, files in os.walk(sub_dir):
            for file in files:
                if file.endswith('.py'):
                    subprocess.run(['python', os.path.join(dirpath, file)])
                elif file.endswith('.js'):
                    subprocess.run(['node', os.path.join(dirpath, file)])
    log_message("Models/Utils executed")

def run_tests(root, run_tests=False):
    if not run_tests:
        return
    test_dir = os.path.join(root, 'tests')
    for dirpath, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                subprocess.run(['pytest', os.path.join(dirpath, file)])
    log_message("Tests run")

async def start_api_frontend(root):
    api_dir = os.path.join(root, 'src/api')
    frontend_dir = os.path.join(root, 'src/frontend')
    if os.path.exists(os.path.join(api_dir, 'main.py')):
        subprocess.Popen(['python', os.path.join(api_dir, 'main.py')])
    if os.path.exists(os.path.join(frontend_dir, 'vite.config.js')):
        subprocess.Popen(['npx', 'vite', 'dev'], cwd=frontend_dir)
    log_message("API/Frontend started")

def deploy(root, deploy_to_modal=False):
    if not deploy_to_modal:
        return
    deploy_dir = os.path.join(root, 'deploy/modal')
    for dirpath, _, files in os.walk(deploy_dir):
        for file in files:
            if file.endswith('.py'):
                subprocess.run(['modal', 'deploy', os.path.join(dirpath, file)])
    docker_dir = os.path.join(root, 'deploy/docker')
    if os.path.exists(os.path.join(docker_dir, 'docker-compose.yml')):
        subprocess.run(['docker-compose', 'up', '-d'], cwd=docker_dir)
    log_message("Deployed to Modal/Docker")

def monitor_logs(root):
    log_dir = os.path.join(root, 'logs')
    log_file = os.path.join(log_dir, 'app.log')
    if os.path.exists(log_file):
        subprocess.Popen(['tail', '-f', log_file])
    log_message("Logs monitoring started")

async def main(root, deploy_to_modal, run_tests, verbose):
    await init_hermes()
    configs = load_configs(root)
    prep_data(root)
    exec_notebooks(root)
    exec_models_utils(root)
    run_tests(root, run_tests)
    await start_api_frontend(root)
    deploy(root, deploy_to_modal)
    monitor_logs(root)
    log_message("Bootup Complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy-to-modal', action='store_true')
    parser.add_argument('--run-tests', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asyncio.run(main(root, args.deploy_to_modal, args.run_tests, args.verbose))