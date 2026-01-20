import os
import subprocess
import json
import boto3
from botocore.exceptions import ClientError
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import google.auth

def check_aws_credentials():
    """Check for AWS credentials in IAM or Secrets Manager."""
    try:
        # Try IAM credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials and credentials.access_key:
            return {
                "AWS_ACCESS_KEY": credentials.access_key,
                "AWS_SECRET_KEY": credentials.secret_key
            }
    except Exception:
        pass

    # Try Secrets Manager
    try:
        client = boto3.client("secretsmanager", region_name="us-east-1")
        secret = client.get_secret_value(SecretId="nexus-backup")
        secret_dict = json.loads(secret["SecretString"])
        return {
            "AWS_ACCESS_KEY": secret_dict["AWS_ACCESS_KEY"],
            "AWS_SECRET_KEY": secret_dict["AWS_SECRET_KEY"]
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            print("AWS access denied. Attempting SSO login...")
            return aws_sso_login()
        raise e

def aws_sso_login():
    """Prompt for AWS SSO login and retrieve temporary credentials."""
    try:
        subprocess.run(
            ["aws", "sso", "login", "--profile", "nexus-profile"],
            check=True,
            capture_output=True
        )
        print("AWS SSO login successful.")
        session = boto3.Session(profile_name="nexus-profile")
        credentials = session.get_credentials()
        return {
            "AWS_ACCESS_KEY": credentials.access_key,
            "AWS_SECRET_KEY": credentials.secret_key
        }
    except subprocess.CalledProcessError:
        print("AWS SSO login failed. Please configure SSO: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html")
        return None

def check_google_credentials(project_id="nova-prime-genesis"):
    """Check Google Cloud ADC and prompt for login if needed."""
    try:
        credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if credentials.valid:
            return {"HF_TOKEN": os.getenv("HF_TOKEN", "your_hf_token")}
        credentials.refresh(Request())
        return {"HF_TOKEN": os.getenv("HF_TOKEN", "your_hf_token")}
    except DefaultCredentialsError:
        print("No valid Google Cloud credentials. Running gcloud login...")
        subprocess.run(
            ["gcloud", "auth", "application-default", "login", "--project", project_id],
            check=True,
            capture_output=True
        )
        print("Google Cloud login successful.")
        return {"HF_TOKEN": os.getenv("HF_TOKEN", "your_hf_token")}

def write_env_file(aws_creds, gcp_creds):
    """Write credentials to .env file."""
    env_path = "C:\\Nexus\\deploy\\.env"
    with open(env_path, "w") as f:
        if aws_creds:
            f.write(f"AWS_ACCESS_KEY={aws_creds['AWS_ACCESS_KEY']}\n")
            f.write(f"AWS_SECRET_KEY={aws_creds['AWS_SECRET_KEY']}\n")
        if gcp_creds:
            f.write(f"HF_TOKEN={gcp_creds['HF_TOKEN']}\n")
    print(f"Credentials written to {env_path}")

def main():
    print("Pulling credentials for Lillith deployment...")
    aws_creds = check_aws_credentials()
    gcp_creds = check_google_credentials()
    if aws_creds and gcp_creds:
        write_env_file(aws_creds, gcp_creds)
        print("Credentials ready. Proceed with 'docker-compose up -d' in C:\\Nexus\\deploy")
    else:
        print("Failed to retrieve credentials. Check AWS SSO or gcloud setup.")

if __name__ == "__main__":
    main()