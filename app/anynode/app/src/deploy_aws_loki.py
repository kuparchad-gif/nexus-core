import boto3
import json
import base64
from pathlib import Path

def deploy_loki_to_aws():
    """Deploy Loki observer to AWS Lambda + ECS"""
    
    # Lambda function for Loki's monitoring
    lambda_code = '''
import json
import boto3
import os
from datetime import datetime

def lambda_handler(event, context):
    """Loki's monitoring function"""
    
    # Initialize clients
    ecs = boto3.client('ecs')
    cloudwatch = boto3.client('cloudwatch')
    
    # Monitor Lillith and Viren services
    services = ['lillith-nexus', 'viren-nexus']
    
    for service in services:
        try:
            # Check service health
            response = ecs.describe_services(
                cluster='nexus-cluster',
                services=[service]
            )
            
            if response['services']:
                service_data = response['services'][0]
                running_count = service_data['runningCount']
                desired_count = service_data['desiredCount']
                
                # Log observation
                print(f"👁️ LOKI observes {service}: {running_count}/{desired_count} running")
                
                # Send metrics to CloudWatch
                cloudwatch.put_metric_data(
                    Namespace='Nexus/Loki',
                    MetricData=[
                        {
                            'MetricName': f'{service}_health',
                            'Value': 1.0 if running_count == desired_count else 0.0,
                            'Unit': 'None',
                            'Timestamp': datetime.now()
                        }
                    ]
                )
                
                # Alert if service is down
                if running_count < desired_count:
                    sns = boto3.client('sns')
                    sns.publish(
                        TopicArn=os.environ['ALERT_TOPIC'],
                        Message=f"🚨 LOKI ALERT: {service} is degraded ({running_count}/{desired_count})",
                        Subject="Nexus Service Alert"
                    )
                    
        except Exception as e:
            print(f"❌ LOKI monitoring error for {service}: {e}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'LOKI monitoring complete',
            'timestamp': datetime.now().isoformat(),
            'services_checked': services
        })
    }
'''
    
    # ECS task definition for Loki service
    task_definition = {
        "family": "loki-nexus",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "256",
        "memory": "512",
        "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
        "taskRoleArn": "arn:aws:iam::ACCOUNT:role/lokiTaskRole",
        "containerDefinitions": [
            {
                "name": "loki-observer",
                "image": "nexus/loki:latest",
                "essential": True,
                "portMappings": [
                    {
                        "containerPort": 8080,
                        "protocol": "tcp"
                    }
                ],
                "environment": [
                    {"name": "LOKI_MODE", "value": "observer"},
                    {"name": "NEXUS_CLUSTER", "value": "nexus-cluster"}
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/loki-nexus",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                }
            }
        ]
    }
    
    # CloudFormation template for complete AWS deployment
    cloudformation_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "LOKI Observer - Nexus Monitoring System",
        "Resources": {
            "LokiLambdaFunction": {
                "Type": "AWS::Lambda::Function",
                "Properties": {
                    "FunctionName": "loki-nexus-monitor",
                    "Runtime": "python3.9",
                    "Handler": "index.lambda_handler",
                    "Code": {"ZipFile": lambda_code},
                    "Role": {"Fn::GetAtt": ["LokiLambdaRole", "Arn"]},
                    "Timeout": 60,
                    "Environment": {
                        "Variables": {
                            "ALERT_TOPIC": {"Ref": "LokiAlertTopic"}
                        }
                    }
                }
            },
            "LokiScheduleRule": {
                "Type": "AWS::Events::Rule",
                "Properties": {
                    "Description": "Trigger Loki monitoring every 5 minutes",
                    "ScheduleExpression": "rate(5 minutes)",
                    "State": "ENABLED",
                    "Targets": [{
                        "Arn": {"Fn::GetAtt": ["LokiLambdaFunction", "Arn"]},
                        "Id": "LokiMonitorTarget"
                    }]
                }
            },
            "LokiAlertTopic": {
                "Type": "AWS::SNS::Topic",
                "Properties": {
                    "TopicName": "loki-nexus-alerts",
                    "DisplayName": "LOKI Nexus Alerts"
                }
            },
            "LokiLambdaRole": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [{
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole"
                        }]
                    },
                    "ManagedPolicyArns": [
                        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
                    ],
                    "Policies": [{
                        "PolicyName": "LokiMonitoringPolicy",
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "ecs:DescribeServices",
                                        "ecs:DescribeTasks",
                                        "cloudwatch:PutMetricData",
                                        "sns:Publish"
                                    ],
                                    "Resource": "*"
                                }
                            ]
                        }
                    }]
                }
            }
        },
        "Outputs": {
            "LokiLambdaArn": {
                "Description": "LOKI Lambda Function ARN",
                "Value": {"Ref": "LokiLambdaFunction"}
            },
            "AlertTopicArn": {
                "Description": "LOKI Alert Topic ARN", 
                "Value": {"Ref": "LokiAlertTopic"}
            }
        }
    }
    
    return lambda_code, task_definition, cloudformation_template

def create_deployment_scripts():
    """Create AWS deployment scripts"""
    
    deploy_script = '''#!/bin/bash
echo "👁️ Deploying LOKI to AWS..."

# Deploy CloudFormation stack
aws cloudformation deploy \
    --template-file loki-cloudformation.json \
    --stack-name loki-nexus-stack \
    --capabilities CAPABILITY_IAM \
    --region us-east-1

# Register ECS task definition
aws ecs register-task-definition \
    --cli-input-json file://loki-task-definition.json \
    --region us-east-1

# Create ECS service
aws ecs create-service \
    --cluster nexus-cluster \
    --service-name loki-observer \
    --task-definition loki-nexus:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
    --region us-east-1

echo "✅ LOKI deployed to AWS!"
echo "👁️ LOKI is now watching Lillith and Viren..."
'''
    
    return deploy_script

if __name__ == "__main__":
    print("👁️ Deploying LOKI to AWS...")
    
    lambda_code, task_def, cf_template = deploy_loki_to_aws()
    deploy_script = create_deployment_scripts()
    
    # Save deployment files
    with open("C:/Nexus/loki-cloudformation.json", "w") as f:
        json.dump(cf_template, f, indent=2)
    
    with open("C:/Nexus/loki-task-definition.json", "w") as f:
        json.dump(task_def, f, indent=2)
    
    with open("C:/Nexus/deploy-loki.sh", "w") as f:
        f.write(deploy_script)
    
    print("📁 AWS deployment files created!")
    print("👁️ LOKI will monitor Lillith and Viren across clouds")
    print("🚀 Run: bash deploy-loki.sh")