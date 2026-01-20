import subprocess
import json
import boto3
from botocore.exceptions import ClientError
import requests
import uuid

def set_modal():
    """Set Modal profile."""
    subprocess.run(["modal", "config", "set-profile", "aethereal-nexus"], check=True)

def get_viren_envs():
    """List Viren-DB0 to Viren-DB7."""
    return [f"Viren-DB{i}" for i in range(8)]

def deploy_modal():
    """Deploy Modal environments with storage and ANYNODE."""
    for env in get_viren_envs():
        subprocess.run(["modal", "volume", "create", f"{env.lower()}-db"], check=True)
        with open("C:\\AetherealNexus\\modal.yaml", "r") as f:
            cfg = f.read().replace("VIRAN_ENV_NAME", env)
        with open(f"C:\\AetherealNexus\\modal-{env}.yaml", "w") as f:
            f.write(cfg)
        subprocess.run(["modal", "config", "set-environment", env], check=True)
        subprocess.run(["modal", "deploy", "--config", f"C:\\AetherealNexus\\modal-{env}.yaml"], check=True)
        # Deploy ANYNODE
        anynode_id = f"anynode-{env.lower()}"
        subprocess.run([
            "modal", "run", "--cpu", "2", "--memory", "8", "--name", anynode_id,
            "python", "-m", "lilith_swarm_core_deploy", "--node-type", "anynode"
        ], check=True)
        with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
            phone_dir = json.load(f)
            for i in range(1, 7):
                phone_dir['phone_dir']['services'].append({
                    'name': f'viren-{env.lower()}-{i}',
                    'endpoint': f'https://{env.lower()}-{i}.modal.run',
                    'type': 'viren_service',
                    'connected': True,
                    'storage': f'{env.lower()}-db'
                })
            phone_dir['phone_dir']['services'].append({
                'name': anynode_id,
                'endpoint': f'https://{anynode_id}.modal.run',
                'type': 'anynode',
                'connected': True
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
        subprocess.run([
            "consul", "service", "register", "-name", f"anynode-{env.lower()}",
            "-address", f"{anynode_id}.modal.run", "-port", "8081",
            "-meta", f"env={env},type=anynode", "-id", anynode_id
        ], check=True)
        print(f"Deployed {env} with ANYNODE")

def create_aws_account(org_client, name, email):
    """Create AWS sub-account."""
    try:
        resp = org_client.create_account(Email=email, AccountName=name)
        return resp['CreateAccountStatus']['AccountId']
    except ClientError:
        return None

def deploy_aws():
    """Deploy AWS free-tier services with ANYNODEs."""
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
                        'environment': {'CPU': '2', 'MEMORY': '8Gi'}
                    }},
                    publicEndpoint={'containerName': 'lilith-core', 'containerPort': 8081}
                )
                # Deploy ANYNODE
                ls.create_container_service(
                    serviceName=f'anynode-{acc["name"]}',
                    power='micro',
                    scale=1,
                    publicDomainNames={f'anynode-{acc["name"]}': [f'anynode-{acc["name"]}.local']},
                    containers={'anynode': {
                        'image': 'gcr.io/nexus-core-prod/lilith-core',
                        'ports': {'8081': 'HTTP'},
                        'command': ['python', '-m', 'lilith_swarm_core_deploy', '--node-type', 'anynode'],
                        'environment': {'CPU': '2', 'MEMORY': '8Gi'}
                    }},
                    publicEndpoint={'containerName': 'anynode', 'containerPort': 8081}
                )
                with open("C:\\AetherealNexus\\phone_directory.json", 'r+') as f:
                    phone_dir = json.load(f)
                    for i in range(1, 7):
                        phone_dir['phone_dir']['services'].append({
                            'name': f'lilith-{acc["name"]}-{region}-{i}',
                            'endpoint': f'http://lilith-{acc["name"]}-{i}.{region}.lightsail.local:8081',
                            'type': 'aws_service',
                            'connected': True
                        })
                    phone_dir['phone_dir']['services'].append({
                        'name': f'anynode-{acc["name"]}-{region}',
                        'endpoint': f'http://anynode-{acc["name"]}.{region}.lightsail.local:8081',
                        'type': 'anynode',
                        'connected': True
                    })
                    f.seek(0)
                    json.dump(phone_dir, f)
                requests.post("http://localhost:5000/api/config/update", json=phone_dir)
                for i in range(1, 7):
                    subprocess.run([
                        "consul", "service", "register", "-name", f"lilith-{acc["name"]}",
                        "-address", f"lilith-{acc["name"]}-{i}.{region}.lightsail.local", "-port", "8081",
                        "-meta", f"project={acc["name"]},region={region},inst={i}",
                        "-id", f"lilith-{acc["name"]}-{region}-{i}"
                    ], check=True)
                subprocess.run([
                    "consul", "service", "register", "-name", f"anynode-{acc["name"]}",
                    "-address", f"anynode-{acc["name"]}.{region}.lightsail.local", "-port", "8081",
                    "-meta", f"project={acc["name"]},region={region},type=anynode",
                    "-id", f"anynode-{acc["name"]}-{region}"
                ], check=True)
                print(f"Deployed {acc["name"]} in {region} with ANYNODE")
        except ClientError as e:
            print(f"Role error for {acc["name"]}: {e}")

def main():
    set_modal()
    deploy_modal()
    deploy_aws()
    requests.post("http://localhost:5000/api/shell", json={"command": "status"})
    print("Lilith deployed with ANYNODEs!")

if __name__ == "__main__":
    main()