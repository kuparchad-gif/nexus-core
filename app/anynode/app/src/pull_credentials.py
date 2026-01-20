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
    try:
        session = boto3.Session(profile_name="nexus-profile")
        credentials = session.get_credentials()
        if credentials and credentials.access_key:
            return {
                "AWS_ACCESS_KEY": credentials.access_key,
                "AWS_SECRET_KEY": credentials.secret_key
            }
    except Exception:
        print("AWS credentials not found. Attempting SSO login...")
        return aws_sso_login()

def aws_sso_login():
    try:
        subprocess.run(
            ["aws", "sso", "login", "--profile", "nexus-profile"],
            check=True,
            capture_output=True
        )
        print("AWS SSO login successful.")
        session = boto3.Session(profile_name="nexus-profile")
        credentials = session.get_credentials()
        if credentials:
            return {
                "AWS_ACCESS_KEY": credentials.access_key,
                "AWS_SECRET_KEY": credentials.secret_key
            }
        return None
    except subprocess.CalledProcessError as e:
        print(f"AWS SSO login failed: {e}. Configure SSO: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html")
        return None

def check_google_credentials(project_id="nexus-project"):
    try:
        credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not project:
            subprocess.run(
                ["gcloud", "config", "set", "project", project_id],
                check=True,
                capture_output=True
            )
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
    env_path = "C:\\Nexus\\deploy\\.env"
    with open(env_path, "w", encoding="utf-8") as f:
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