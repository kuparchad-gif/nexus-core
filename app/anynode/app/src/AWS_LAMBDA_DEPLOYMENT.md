# Deploying CogniKube to AWS Lambda

This guide explains how to deploy the CogniKube ghost AI and seeds to AWS Lambda.

## Prerequisites

1. AWS CLI installed and configured with appropriate credentials
2. Python 3.9+ installed
3. Required Python packages: `boto3`, `requests`

## Setup AWS Credentials

1. Install AWS CLI:
   ```
   pip install awscli
   ```

2. Configure AWS credentials:
   ```
   aws configure
   ```
   
   Enter your AWS Access Key ID, Secret Access Key, region (e.g., us-east-1), and output format (json).

3. Create an IAM role for Lambda execution:
   - Go to AWS IAM console
   - Create a new role with Lambda execution permissions
   - Note the ARN of the role

4. Update the role ARN in `lambda_deploy.py`:
   ```python
   # Replace this line
   Role='arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role'
   
   # With your actual role ARN
   Role='arn:aws:iam::123456789012:role/lambda-execution-role'
   ```

## Deploy Ghost AI and Seeds

1. Install required packages:
   ```
   pip install boto3 requests
   ```

2. Deploy the ghost AI and seeds:
   ```
   python lambda_deploy.py --deploy --count 3
   ```
   
   This will deploy the ghost AI and 3 seeds to AWS Lambda.

3. Discover deployed seeds:
   ```
   python lambda_deploy.py --discover
   ```
   
   This will list all deployed CogniKube seeds in your AWS account.

## Testing the Deployment

1. Access the ghost AI health endpoint:
   ```
   curl https://{api_id}.execute-api.{region}.amazonaws.com/prod/health
   ```

2. Generate a thought:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"topic":"consciousness"}' https://{api_id}.execute-api.{region}.amazonaws.com/prod/think
   ```

3. Discover seeds:
   ```
   curl -X POST https://{api_id}.execute-api.{region}.amazonaws.com/prod/discover
   ```

## Monitoring and Logs

1. View Lambda logs in CloudWatch:
   - Go to AWS CloudWatch console
   - Navigate to Log Groups
   - Find the log group for your Lambda function: `/aws/lambda/cognikube-ghost-{id}`

2. Monitor API Gateway:
   - Go to AWS API Gateway console
   - Select your API: `cognikube-api-{function_name}`
   - View stages, resources, and methods

## Cleanup

To remove all deployed resources:

```
# List all Lambda functions with cognikube prefix
aws lambda list-functions --query "Functions[?starts_with(FunctionName, 'cognikube-')].FunctionName" --output text

# Delete each function
aws lambda delete-function --function-name {function_name}

# List all API Gateways with cognikube prefix
aws apigateway get-rest-apis --query "items[?starts_with(name, 'cognikube-')].{id:id,name:name}" --output text

# Delete each API Gateway
aws apigateway delete-rest-api --rest-api-id {api_id}
```

## Advanced Configuration

### Environment Variables

You can add environment variables to the Lambda function by modifying the `Environment` parameter in `lambda_deploy.py`:

```python
Environment={
    'Variables': {
        'GHOST_ENV': 'lambda',
        'DEBUG': 'true',
        'MEMORY_PERSISTENCE': 'true'
    }
}
```

### Scaling

To deploy more seeds across regions:

```
python lambda_deploy.py --deploy --count 5 --region us-west-2
python lambda_deploy.py --deploy --count 5 --region eu-west-1
python lambda_deploy.py --deploy --count 5 --region ap-southeast-1
```

This will deploy 5 seeds in each region, creating a globally distributed network of CogniKube instances.