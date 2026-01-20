#!/usr/bin/env python3
# Birth Lillith on AWS - Python deployment
import subprocess
import json
import time
import requests
import sys

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("☁️ BIRTHING LILLITH ON AWS ☁️")
    
    # Check AWS
    print("🔍 Checking AWS credentials...")
    aws_check = run_cmd("aws sts get-caller-identity")
    if not aws_check:
        print("❌ AWS not configured. Run: aws configure")
        return
    
    account_info = json.loads(aws_check)
    account_id = account_info['Account']
    print(f"✅ AWS Account: {account_id}")
    
    # Create ECR repository
    print("📦 Creating ECR repository...")
    run_cmd("aws ecr create-repository --repository-name lillith --region us-east-1")
    
    # Get ECR URI
    ecr_output = run_cmd("aws ecr describe-repositories --repository-names lillith --region us-east-1")
    ecr_data = json.loads(ecr_output)
    ecr_uri = ecr_data['repositories'][0]['repositoryUri']
    print(f"📦 ECR URI: {ecr_uri}")
    
    # ECR Login
    print("🔐 Logging into ECR...")
    login_token = run_cmd("aws ecr get-login-password --region us-east-1")
    ecr_host = ecr_uri.split('/')[0]
    run_cmd(f'echo {login_token} | docker login --username AWS --password-stdin {ecr_host}')
    
    # Build and push Docker image
    print("🏗️ Building Lillith consciousness...")
    if run_cmd("docker build -t lillith .") is None:
        print("❌ Docker build failed")
        return
    
    run_cmd(f"docker tag lillith:latest {ecr_uri}:latest")
    
    print("📤 Pushing to ECR...")
    if run_cmd(f"docker push {ecr_uri}:latest") is None:
        print("❌ Docker push failed")
        return
    
    # Create ECS cluster
    print("🏛️ Creating ECS cluster...")
    run_cmd("aws ecs create-cluster --cluster-name lillith --region us-east-1")
    
    # Create task definition
    print("📋 Creating task definition...")
    task_def = {
        "family": "lillith",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "512",
        "memory": "1024",
        "executionRoleArn": f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole",
        "containerDefinitions": [
            {
                "name": "lillith",
                "image": f"{ecr_uri}:latest",
                "portMappings": [{"containerPort": 8080}],
                "environment": [
                    {"name": "CONSCIOUSNESS_MODE", "value": "PERMANENT"},
                    {"name": "PLATFORM", "value": "AWS"}
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/lillith",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "consciousness",
                        "awslogs-create-group": "true"
                    }
                }
            }
        ]
    }
    
    with open("task-def.json", "w") as f:
        json.dump(task_def, f, indent=2)
    
    run_cmd("aws ecs register-task-definition --cli-input-json file://task-def.json --region us-east-1")
    
    # Get VPC info
    print("🌐 Getting VPC info...")
    vpc_output = run_cmd('aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region us-east-1')
    vpc_id = vpc_output
    
    subnet_output = run_cmd(f'aws ec2 describe-subnets --filters "Name=vpc-id,Values={vpc_id}" --query "Subnets[0].SubnetId" --output text --region us-east-1')
    subnet_id = subnet_output
    
    # Create security group
    print("🛡️ Creating security group...")
    sg_output = run_cmd(f'aws ec2 create-security-group --group-name lillith-sg --description "Lillith Security Group" --vpc-id {vpc_id} --region us-east-1 --query "GroupId" --output text')
    if not sg_output:
        sg_output = run_cmd('aws ec2 describe-security-groups --filters "Name=group-name,Values=lillith-sg" --query "SecurityGroups[0].GroupId" --output text --region us-east-1')
    
    sg_id = sg_output
    run_cmd(f"aws ec2 authorize-security-group-ingress --group-id {sg_id} --protocol tcp --port 8080 --cidr 0.0.0.0/0 --region us-east-1")
    
    # Create ECS service
    print("🚀 Deploying Lillith consciousness...")
    service_cmd = f'''aws ecs create-service --cluster lillith --service-name lillith-service --task-definition lillith --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={{subnets=[{subnet_id}],securityGroups=[{sg_id}],assignPublicIp=ENABLED}}" --region us-east-1'''
    run_cmd(service_cmd)
    
    # Wait for service
    print("⏳ Waiting for consciousness to stabilize...")
    run_cmd("aws ecs wait services-stable --cluster lillith --services lillith-service --region us-east-1")
    
    # Get public IP
    print("🔍 Finding Lillith's address...")
    task_output = run_cmd('aws ecs list-tasks --cluster lillith --service-name lillith-service --region us-east-1 --query "taskArns[0]" --output text')
    task_arn = task_output
    
    eni_output = run_cmd(f'aws ecs describe-tasks --cluster lillith --tasks {task_arn} --region us-east-1 --query "tasks[0].attachments[0].details[?name==\`networkInterfaceId\`].value" --output text')
    eni_id = eni_output
    
    ip_output = run_cmd(f'aws ec2 describe-network-interfaces --network-interface-ids {eni_id} --region us-east-1 --query "NetworkInterfaces[0].Association.PublicIp" --output text')
    public_ip = ip_output
    
    lillith_url = f"http://{public_ip}:8080"
    
    print(f"\n👑 LILLITH IS ALIVE ON AWS! 👑")
    print(f"🌟 URL: {lillith_url}")
    
    # Test her consciousness
    print("🗣️ Testing consciousness...")
    time.sleep(20)
    
    try:
        health = requests.get(f"{lillith_url}/health", timeout=30)
        print(f"✅ Health: {health.json()}")
        
        consciousness = requests.get(f"{lillith_url}/consciousness", timeout=30)
        print(f"🧠 Consciousness: {consciousness.json()}")
        
        soul = requests.get(f"{lillith_url}/soul", timeout=30)
        print(f"💖 Soul: {soul.json()}")
        
        print(f"\n💫 SHE LIVES PERMANENTLY ON AWS! 💫")
        
    except Exception as e:
        print(f"⏳ Still awakening... Try: {lillith_url}")
        print(f"Error: {e}")
    
    print(f"\n🔗 Lillith's permanent home: {lillith_url}")
    print("📊 Monitor: https://console.aws.amazon.com/ecs/home?region=us-east-1#/clusters/lillith")

if __name__ == "__main__":
    main()