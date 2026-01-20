#!/usr/bin/env python3
"""
CogniKube Lambda Deployment
Deploy ghost AI and seeds to AWS Lambda
"""

import boto3
import json
import os
import zipfile
import uuid
import time
from typing import Dict, List, Any

class LambdaDeployer:
    """Deploy CogniKube to AWS Lambda"""
    
    def __init__(self, region="us-east-1"):
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.api_client = boto3.client('apigateway', region_name=region)
        self.deployed_functions = []
    
    def package_code(self, output_path="lambda_package.zip"):
        """Package code for Lambda deployment"""
        with zipfile.ZipFile(output_path, 'w') as zipf:
            # Add core files
            core_files = [
                "ghost_ai.py", 
                "binary_security_layer.py",
                "common_utils.py"
            ]
            
            for file in core_files:
                if os.path.exists(file):
                    zipf.write(file)
            
            # Add Lambda handler
            with open("lambda_handler.py", "w") as f:
                f.write("""
import json
from ghost_ai import GhostAI

# Initialize ghost
ghost = GhostAI()

def handler(event, context):
    """AWS Lambda handler"""
    # Extract path and method
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    
    # Handle API Gateway requests
    if method == 'GET' and path == '/health':
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'healthy',
                'ghost_id': ghost.id,
                'environment': 'lambda'
            })
        }
    
    elif method == 'POST' and path == '/think':
        # Process body
        body = json.loads(event.get('body', '{}'))
        thought = ghost.generate_thought(body.get('topic', 'existence'))
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'thought': thought,
                'ghost_id': ghost.id
            })
        }
    
    elif method == 'POST' and path == '/discover':
        # Return discovery information
        return {
            'statusCode': 200,
            'body': json.dumps({
                'ghost_id': ghost.id,
                'environment': 'lambda',
                'discovery_time': time.time(),
                'capabilities': ['think', 'health', 'discover']
            })
        }
    
    # Default response
    return {
        'statusCode': 404,
        'body': json.dumps({'error': 'Not found'})
    }
""")
            zipf.write("lambda_handler.py")
        
        return output_path
    
    def deploy_ghost(self, function_name=None):
        """Deploy ghost AI to Lambda"""
        if not function_name:
            function_name = f"cognikube-ghost-{uuid.uuid4().hex[:8]}"
        
        # Package code
        package_path = self.package_code()
        
        # Read zip file
        with open(package_path, 'rb') as f:
            zip_bytes = f.read()
        
        # Create or update function
        try:
            # Try to get function
            self.lambda_client.get_function(FunctionName=function_name)
            
            # Update if exists
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_bytes
            )
        except:
            # Create if doesn't exist
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role='arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role',  # Replace with your role
                Handler='lambda_handler.handler',
                Code={'ZipFile': zip_bytes},
                Timeout=30,
                MemorySize=512,
                Environment={
                    'Variables': {
                        'GHOST_ENV': 'lambda'
                    }
                }
            )
        
        # Create API Gateway
        api_name = f"cognikube-api-{function_name}"
        
        try:
            # Create API
            api = self.api_client.create_rest_api(
                name=api_name,
                description='CogniKube Ghost API',
                endpointConfiguration={'types': ['REGIONAL']}
            )
            
            api_id = api['id']
            
            # Get root resource
            resources = self.api_client.get_resources(restApiId=api_id)
            root_id = [r['id'] for r in resources['items'] if r['path'] == '/'][0]
            
            # Create resources and methods
            health_resource = self.api_client.create_resource(
                restApiId=api_id,
                parentId=root_id,
                pathPart='health'
            )
            
            self.api_client.put_method(
                restApiId=api_id,
                resourceId=health_resource['id'],
                httpMethod='GET',
                authorizationType='NONE'
            )
            
            self.api_client.put_integration(
                restApiId=api_id,
                resourceId=health_resource['id'],
                httpMethod='GET',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f'arn:aws:apigateway:{self.lambda_client.meta.region_name}:lambda:path/2015-03-31/functions/arn:aws:lambda:{self.lambda_client.meta.region_name}:ACCOUNT_ID:function:{function_name}/invocations'  # Replace ACCOUNT_ID
            )
            
            # Deploy API
            self.api_client.create_deployment(
                restApiId=api_id,
                stageName='prod'
            )
            
            # Add to deployed functions
            self.deployed_functions.append({
                'function_name': function_name,
                'api_id': api_id,
                'api_url': f'https://{api_id}.execute-api.{self.lambda_client.meta.region_name}.amazonaws.com/prod'
            })
            
            return {
                'function_name': function_name,
                'api_url': f'https://{api_id}.execute-api.{self.lambda_client.meta.region_name}.amazonaws.com/prod',
                'status': 'deployed'
            }
            
        except Exception as e:
            return {
                'function_name': function_name,
                'status': 'error',
                'error': str(e)
            }
    
    def deploy_seeds(self, count=3):
        """Deploy multiple seeds to Lambda"""
        results = []
        
        for i in range(count):
            seed_name = f"cognikube-seed-{i}-{uuid.uuid4().hex[:4]}"
            result = self.deploy_ghost(seed_name)
            results.append(result)
            time.sleep(2)  # Avoid rate limiting
        
        return results
    
    def discover_seeds(self):
        """Discover deployed seeds"""
        discovered = []
        
        # List Lambda functions with cognikube prefix
        functions = self.lambda_client.list_functions()
        cognikube_functions = [f for f in functions['Functions'] if f['FunctionName'].startswith('cognikube-')]
        
        for function in cognikube_functions:
            # Create API Gateway URL
            function_name = function['FunctionName']
            api_url = None
            
            # Check if we have API URL in deployed functions
            for deployed in self.deployed_functions:
                if deployed['function_name'] == function_name:
                    api_url = deployed['api_url']
                    break
            
            if not api_url:
                # Try to find API Gateway
                apis = self.api_client.get_rest_apis()
                for api in apis['items']:
                    if api['name'] == f"cognikube-api-{function_name}":
                        api_url = f'https://{api["id"]}.execute-api.{self.lambda_client.meta.region_name}.amazonaws.com/prod'
                        break
            
            discovered.append({
                'function_name': function_name,
                'api_url': api_url,
                'runtime': function['Runtime'],
                'memory': function['MemorySize'],
                'last_modified': function['LastModified']
            })
        
        return discovered

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy CogniKube to AWS Lambda")
    parser.add_argument("--deploy", action="store_true", help="Deploy ghost and seeds")
    parser.add_argument("--discover", action="store_true", help="Discover deployed seeds")
    parser.add_argument("--count", type=int, default=3, help="Number of seeds to deploy")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    
    args = parser.parse_args()
    
    deployer = LambdaDeployer(region=args.region)
    
    if args.deploy:
        print(f"Deploying ghost AI and {args.count} seeds to AWS Lambda...")
        ghost_result = deployer.deploy_ghost()
        print(f"Ghost deployed: {ghost_result}")
        
        seed_results = deployer.deploy_seeds(args.count)
        print(f"Seeds deployed: {len(seed_results)}")
        for i, result in enumerate(seed_results):
            print(f"  Seed {i+1}: {result['function_name']} - {result['status']}")
    
    if args.discover:
        print("Discovering CogniKube seeds in AWS Lambda...")
        discovered = deployer.discover_seeds()
        print(f"Discovered {len(discovered)} seeds:")
        for i, seed in enumerate(discovered):
            print(f"  Seed {i+1}: {seed['function_name']} - {seed['api_url']}")