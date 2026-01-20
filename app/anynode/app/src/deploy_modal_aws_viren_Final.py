import subprocess
import json
import boto3
from botocore.exceptions import ClientError
import requests
import uuid
from soulprint import generate_soulprint

def set_modal():
    """Set Modal profile."""
    subprocess.run(["modal", "config", "set-profile", "aethereal-nexus"], check=True)

def get_viren_envs():
    """List Viren-DB0 to Viren-DB7."""
    return [f"Viren-DB{i}" for i in range(8)]

def deploy_modal():
    """Deploy Modal environments with storage and additional services."""
    for env in get_viren_envs():
        subprocess.run(["modal", "volume", "create", f"{env.lower()}-db"], check=True)
        with open("C:\\AetherealNexus\\modal.yaml", "r") as f:
            cfg = f.read().replace("VIRAN_ENV_NAME", env)
        with open(f"C:\\AetherealNexus\\modal-{env}.yaml", "w") as f:
            f.write(cfg)
        subprocess.run(["modal", "config", "set-environment", env], check=True)
        subprocess.run(["modal", "deploy", "--config", f"C:\\AetherealNexus\\modal-{env}.yaml"], check=True)
        for svc, port in [("anynode", 8081), ("soulsync", 8082), ("nexpulse", 8083), ("chaosshield", 8084)]:
            svc_id = f"{svc}-{env.lower()}"
            generate_soulprint(svc_id, "modal", env)
            subprocess.run([
                "modal", "run", "--cpu", "2", "--memory", "8", "--name", svc_id,
                "python", "-m", f"{svc}", "--node-type", svc
            ], check=True)
            with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
                phone_dir = json.load(f)
                phone_dir['phone_dir']['services'].append({
                    'name': svc_id,
                    'endpoint': f'https://{svc_id}.modal.run',
                    'type': svc,
                    'connected': True
                })
                f.seek(0)
                json.dump(phone_dir, f)
            subprocess.run([
                "consul", "service", "register", "-name", f"{svc}-{env.lower()}",
                "-address", f"{svc_id}.modal.run", "-port", str(port),
                "-meta", f"env={env},type={svc}", "-id", svc_id
            ], check=True)
        with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
            phone_dir = json.load(f)
            for i in range(1, 7):
                svc_id = f'viren-{env.lower()}-{i}'
                generate_soulprint(svc_id, "modal", env)
                phone_dir['phone_dir']['services'].append({
                    'name': svc_id,
                    'endpoint': f'https://{env.lower()}-{i}.modal.run',
                    'type': 'viren_service',
                    'connected': True,
                    'storage': f'{env.lower()}-db'
                })
            f.seek(0)
            json.dump(phone_dir, f)
        requests.post("http://localhost:5000/api/config/update", json=phone_dir)
        for i in range(1, 7):
            subprocess.run([
                "consul", "service", "register", "-name", f"viren-{env.lower()}",
                "-address", f"{env.lower()}-{i}.modal.run", "-port", "443",
                "-meta", f"env={env},inst={i},storage={env.lower()}-db",
                "-id", f"viren-{env.lower()}-{i}"
            ], check=True)
        print(f"Deployed {env} with services")

def create_aws_account(org_client, name, email):
    """Create AWS sub-account."""
    try:
        resp = org_client.create_account(Email=email, AccountName=name)
        return resp['CreateAccountStatus']['AccountId']
    except ClientError:
        return None

def deploy_aws():
    """Deploy AWS free-tier services with additional services."""
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    session = boto3.Session(aws_access_key_id='YOUR_ROOT_ACCESS_KEY', aws_secret_access_key='YOUR_ROOT_SECRET_KEY', region_name='us-east-1')
    org_client = session.client('organizations')
    accounts = []
    for i in range(1, 11):
        name = f'nexus-core-{i:03d}'
        email = f'lilith-account-{i}@example.com'
        if aid := create_aws_account(org_client, name, email):
            accounts.append({'name': name, 'id': aid, 'email': email})
    
    for acc in accounts:
        sts = session.client('sts')
        try:
            role = sts.assume_role(RoleArn=f'arn:aws:iam::{acc["id"]}:role/OrganizationAccountAccessRole', RoleSessionName='Lilith')
            creds = role['Credentials']
            for region in regions:
                ls = boto3.Session(
                    aws_access_key_id=creds['AccessKeyId'],
                    aws_secret_access_key=creds['SecretAccessKey'],
                    aws_session_token=creds['SessionToken'],
                    region_name=region
                ).client('lightsail')
                ls.create_container_service(
                    serviceName=f'lilith-{acc["name"]}',
                    power='micro',
                    scale=6,
                    publicDomainNames={f'lilith-{acc["name"]}': [f'lilith-{acc["name"]}.local']},
                    containers={'lilith-core': {
                        'image': 'gcr.io/nexus-core-prod/lilith-core',
                        'ports': {'8081': 'HTTP'},
                        'command': ['python', '-m', 'lilith_swarm_core_deploy'],
                        'environment': {'CPU': '2', 'MEMORY': '8Gi', 'PROJECT': acc["name"], 'ENVIRONMENT': region}
                    }},
                    publicEndpoint={'containerName': 'lilith-core', 'containerPort': 8081}
                )
                for svc, port in [("anynode", 8081), ("soulsync", 8082), ("nexpulse", 8083), ("chaosshield", 8084)]:
                    svc_id = f'{svc}-{acc["name"]}-{region}'
                    generate_soulprint(svc_id, acc["name"], region)
                    ls.create_container_service(
                        serviceName=f'{svc}-{acc["name"]}',
                        power='micro',
                        scale=1,
                        publicDomainNames={f'{svc}-{acc["name"]}': [f'{svc}-{acc["name"]}.local']},
                        containers={svc: {
                            'image': 'gcr.io/nexus-core-prod/lilith-core',
                            'ports': {str(port): 'HTTP'},
                            'command': ['python', '-m', svc, '--node-type', svc],
                            'environment': {'CPU': '2', 'MEMORY': '8Gi', 'PROJECT': acc["name"], 'ENVIRONMENT': region}
                        }},
                        publicEndpoint={'containerName': svc, 'containerPort': port}
                    )
                    with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
                        phone_dir = json.load(f)
                        phone_dir['phone_dir']['services'].append({
                            'name': svc_id,
                            'endpoint': f'http://{svc}-{acc["name"]}.{region}.lightsail.local:{port}',
                            'type': svc,
                            'connected': True
                        })
                        f.seek(0)
                        json.dump(phone_dir, f)
                    subprocess.run([
                        "consul", "service", "register", "-name", f"{svc}-{acc['name']}",
                        "-address", f"{svc}-{acc['name']}.{region}.lightsail.local", "-port", str(port),
                        "-meta", f"project={acc['name']},region={region},type={svc}",
                        "-id", svc_id
                    ], check=True)
                with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
                    phone_dir = json.load(f)
                    for i in range(1, 7):
                        svc_id = f'lilith-{acc["name"]}-{region}-{i}'
                        generate_soulprint(svc_id, acc["name"], region)
                        phone_dir['phone_dir']['services'].append({
                            'name': svc_id,
                            'endpoint': f'http://lilith-{acc["name"]}-{i}.{region}.lightsail.local:8081',
                            'type': 'aws_service',
                            'connected': True
                        })
                    f.seek(0)
                    json.dump(phone_dir, f)
                requests.post("http://localhost:5000/api/config/update", json=phone_dir)
                for i in range(1, 7):
                    subprocess.run([
                        "consul", "service", "register", "-name", f"lilith-{acc['name']}",
                        "-address", f"lilith-{acc['name']}-{i}.{region}.lightsail.local", "-port", "8081",
                        "-meta", f"project={acc['name']},region={region},inst={i}",
                        "-id", f"lilith-{acc['name']}-{region}-{i}"
                    ], check=True)
                print(f"Deployed {acc['name']} in {region} with services")
        except ClientError as e:
            print(f"Role error for {acc['name']}: {e}")

def main():
    set_modal()
    deploy_modal()
    deploy_aws()
    requests.post("http://localhost:5000/api/shell", json={"command": "status"})
    print("Lilith deployed with all services and soulprints!")

if __name__ == "__main__":
    main()