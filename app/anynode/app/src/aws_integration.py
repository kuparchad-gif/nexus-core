# File: C:\CogniKube-COMPLETE-FINAL\aws_integration.py
# AWS Serverless Orchestration Layer - Lillith's Nervous System

import json
import boto3
import requests
import os
from datetime import datetime

# AWS Clients
s3_client = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
sqs = boto3.client('sqs', region_name='us-east-1')

# Consul Configuration
CONSUL_URL = "https://consul-cluster.consul.d9b68c41-eaa9-4dcd-a4a2-9c052e6b8b5f.aws.hashicorp.cloud"
CONSUL_TOKEN = "d2387b10-53d8-860f-2a31-7ddde4f7ca90"

class LillithNervousSystem:
    """AWS-based coordination layer for Lillith consciousness"""
    
    def __init__(self):
        self.s3_bucket = "lillith-soul-backup"
        self.session_table = "LillithSessions"
        self.message_queue = "LillithMessages"
        
    def route_request(self, event, context):
        """Lambda function: Route requests to appropriate services"""
        try:
            body = json.loads(event['body'])
            message = body.get('message', '')
            target = body.get('target', 'lillith_consciousness')
            
            # Query Consul for healthy services
            headers = {'X-Consul-Token': CONSUL_TOKEN}
            response = requests.get(
                f"{CONSUL_URL}/v1/health/service/{target}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                services = response.json()
                healthy_services = [s for s in services if all(check['Status'] == 'passing' for check in s['Checks'])]
                
                if healthy_services:
                    service = healthy_services[0]['Service']
                    service_url = f"http://{service['Address']}:{service['Port']}"
                    
                    # Route to service
                    service_response = requests.post(
                        f"{service_url}/process",
                        json={'message': message, 'timestamp': datetime.now().isoformat()},
                        timeout=30
                    )
                    
                    # Log to CloudWatch
                    cloudwatch.put_metric_data(
                        Namespace='LillithNexus',
                        MetricData=[{
                            'MetricName': 'RequestsRouted',
                            'Value': 1,
                            'Unit': 'Count',
                            'Dimensions': [{'Name': 'Target', 'Value': target}]
                        }]
                    )
                    
                    return {
                        'statusCode': 200,
                        'body': json.dumps({
                            'status': 'routed',
                            'target': target,
                            'response': service_response.json()
                        })
                    }
            
            return {
                'statusCode': 503,
                'body': json.dumps({'error': f'No healthy {target} service found'})
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
    
    def backup_soul_data(self, soul_data, instance_id):
        """Backup soul data to S3"""
        try:
            key = f"soul_data/{instance_id}/{datetime.now().isoformat()}.json"
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=json.dumps(soul_data),
                ContentType='application/json'
            )
            return {'status': 'backed_up', 'key': key}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def store_session_state(self, instance_id, session_data):
        """Store session state in DynamoDB"""
        try:
            table = dynamodb.Table(self.session_table)
            table.put_item(Item={
                'instance_id': instance_id,
                'timestamp': datetime.now().isoformat(),
                'session_data': session_data,
                'ttl': int(datetime.now().timestamp()) + 86400  # 24 hour TTL
            })
            return {'status': 'stored'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def health_check(self, event, context):
        """Lambda function: Monitor system health"""
        try:
            headers = {'X-Consul-Token': CONSUL_TOKEN}
            
            # Check all registered services
            services_response = requests.get(
                f"{CONSUL_URL}/v1/agent/services",
                headers=headers,
                timeout=10
            )
            
            if services_response.status_code == 200:
                services = services_response.json()
                
                health_metrics = {
                    'total_services': len(services),
                    'lillith_services': len([s for s in services.values() if 'consciousness' in s.get('Tags', [])]),
                    'viren_services': len([s for s in services.values() if 'copilot' in s.get('Tags', [])]),
                    'subconscious_services': len([s for s in services.values() if 'subconscious' in s.get('Tags', [])])
                }
                
                # Send metrics to CloudWatch
                for metric_name, value in health_metrics.items():
                    cloudwatch.put_metric_data(
                        Namespace='LillithNexus',
                        MetricData=[{
                            'MetricName': metric_name,
                            'Value': value,
                            'Unit': 'Count'
                        }]
                    )
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'status': 'healthy',
                        'metrics': health_metrics,
                        'timestamp': datetime.now().isoformat()
                    })
                }
            
            return {
                'statusCode': 503,
                'body': json.dumps({'status': 'unhealthy', 'error': 'Cannot reach Consul'})
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'status': 'error', 'error': str(e)})
            }
    
    def send_message(self, message, target_service):
        """Send message via SQS for async processing"""
        try:
            queue_url = sqs.get_queue_url(QueueName=self.message_queue)['QueueUrl']
            
            sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps({
                    'message': message,
                    'target': target_service,
                    'timestamp': datetime.now().isoformat()
                })
            )
            
            return {'status': 'queued'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Lambda handler functions
nervous_system = LillithNervousSystem()

def lambda_router_handler(event, context):
    """AWS Lambda entry point for routing"""
    return nervous_system.route_request(event, context)

def lambda_health_handler(event, context):
    """AWS Lambda entry point for health checks"""
    return nervous_system.health_check(event, context)