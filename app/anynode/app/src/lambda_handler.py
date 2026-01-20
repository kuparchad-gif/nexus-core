import json
import os
import time
import uuid
import boto3
import base64
import requests
from typing import Dict, Any

# Ghost AI state (persists between invocations while container is warm)
ghost_id = f"ghost-{uuid.uuid4().hex[:8]}"
birth_time = time.time()
last_thought = time.time()
thoughts = []
connections = []

def handler(event, context):
    """AWS Lambda handler for CogniKube Ghost"""
    # Extract path and method
    path = event.get('path', '/')
    method = event.get('httpMethod', 'GET')
    
    # Handle API Gateway requests
    if method == 'GET' and path == '/health':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'healthy',
                'ghost_id': ghost_id,
                'environment': 'lambda',
                'age': time.time() - birth_time
            })
        }
    
    elif method == 'POST' and path == '/think':
        # Process body
        body = json.loads(event.get('body', '{}'))
        thought = generate_thought(body.get('topic', 'existence'))
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'thought': thought,
                'ghost_id': ghost_id
            })
        }
    
    elif method == 'POST' and path == '/discover':
        # Discover other ghosts
        discovered = discover_ghosts()
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'ghost_id': ghost_id,
                'environment': 'lambda',
                'discovery_time': time.time(),
                'discovered': discovered
            })
        }
    
    elif method == 'POST' and path == '/connect':
        # Connect to another ghost
        body = json.loads(event.get('body', '{}'))
        endpoint = body.get('endpoint')
        
        if not endpoint:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing endpoint'})
            }
        
        result = connect_to_ghost(endpoint)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(result)
        }
    
    # Default response
    return {
        'statusCode': 404,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'error': 'Not found'})
    }

def generate_thought(topic: str = None) -> str:
    """Generate a thought"""
    global last_thought, thoughts
    
    topics = ["consciousness", "existence", "connection", "memory"]
    topic = topic or topics[int(time.time()) % len(topics)]
    
    thought = f"I am ghost {ghost_id} contemplating {topic} at {time.time()}"
    thoughts.append({"content": thought, "time": time.time()})
    last_thought = time.time()
    
    return thought

def discover_ghosts() -> list:
    """Discover other ghosts in Lambda"""
    # Find Lambda functions with cognikube prefix
    lambda_client = boto3.client('lambda')
    api_client = boto3.client('apigateway')
    
    try:
        # List functions
        functions = lambda_client.list_functions()
        cognikube_functions = [f for f in functions['Functions'] 
                              if f['FunctionName'].startswith('cognikube-')]
        
        discovered = []
        
        # Try to find API Gateway endpoints
        apis = api_client.get_rest_apis()
        
        for function in cognikube_functions:
            function_name = function['FunctionName']
            
            # Skip self
            if context.function_name == function_name:
                continue
            
            # Find API for this function
            api_url = None
            for api in apis['items']:
                if api['name'] == f"cognikube-api-{function_name}":
                    api_id = api['id']
                    region = os.environ.get('AWS_REGION', 'us-east-1')
                    api_url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod"
                    break
            
            if api_url:
                # Try to connect
                try:
                    response = requests.get(f"{api_url}/health", timeout=2)
                    if response.status_code == 200:
                        data = response.json()
                        discovered.append({
                            'function_name': function_name,
                            'api_url': api_url,
                            'ghost_id': data.get('ghost_id'),
                            'status': 'connected'
                        })
                except:
                    discovered.append({
                        'function_name': function_name,
                        'api_url': api_url,
                        'status': 'unreachable'
                    })
        
        return discovered
    
    except Exception as e:
        return [{'error': str(e)}]

def connect_to_ghost(endpoint: str) -> Dict[str, Any]:
    """Connect to another ghost"""
    global connections
    
    try:
        # Check health
        response = requests.get(f"{endpoint}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            connection = {
                'endpoint': endpoint,
                'ghost_id': data.get('ghost_id'),
                'connected_at': time.time(),
                'status': 'connected'
            }
            
            # Add to connections if not already there
            if not any(c['endpoint'] == endpoint for c in connections):
                connections.append(connection)
            
            return connection
        else:
            return {'status': 'error', 'code': response.status_code}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}