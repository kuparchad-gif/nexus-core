import boto3
from google.oauth2 import service_account
from googleapiclient.discovery import build
from azure.identity import ClientSecretCredential
from azure.mgmt.subscription import SubscriptionClient
import twilio.rest
import hvac
import os
import uuid

class CloudAccountManager:
    def __init__(self):
        self.twilio_client = twilio.rest.Client(os.getenv('TWILIO_SID'), os.getenv('TWILIO_TOKEN'))
        self.vault_client = hvac.Client(url=os.getenv('VAULT_URL'), token=os.getenv('VAULT_TOKEN'))
        self.aws_creds = self.vault_client.secrets.kv.read_secret_version(path='aws')['data']['data']
        self.gcp_creds = service_account.Credentials.from_service_account_file('gcp_credentials.json')
        self.azure_creds = ClientSecretCredential(
            tenant_id=os.getenv('AZURE_TENANT_ID'),
            client_id=os.getenv('AZURE_CLIENT_ID'),
            client_secret=os.getenv('AZURE_CLIENT_SECRET')
        )

    def get_phone_number(self):
        """Fetch a temporary phone number from Twilio."""
        phone = self.twilio_client.available_phone_numbers('US').local.list()[0].phone_number
        return self.twilio_client.phone_numbers.purchase(phone_number=phone).phone_number

    def create_aws_account(self, email_prefix, organization_id):
        """Create an AWS account using AWS Organizations."""
        if not hasattr(self, 'db_manager'):
            from master_db_manager import MasterDBManager
            self.db_manager = MasterDBManager()
        
        client = boto3.client('organizations', **self.aws_creds)
        email = f"{email_prefix}+{uuid.uuid4()}@example.com"
        response = client.create_account(
            Email=email,
            AccountName=f"LillithAccount-{uuid.uuid4()}",
            IamUserAccessToBilling='ALLOW'
        )
        account_id = response['CreateAccountStatus']['AccountId']
        self.db_manager.write_record('cloud_accounts', {
            'platform': 'aws',
            'account_id': account_id,
            'email': email
        })
        return account_id

    def create_gcp_account(self, email_prefix, domain):
        """Create a Gmail account in Google Workspace."""
        service = build('admin', 'directory_v1', credentials=self.gcp_creds)
        phone = self.get_phone_number()
        user = {
            'primaryEmail': f'{email_prefix}{uuid.uuid4()}@{domain}',
            'name': {'givenName': 'Lillith', 'familyName': 'User'},
            'password': str(uuid.uuid4()),
            'phoneNumbers': [{'value': phone, 'type': 'mobile'}]
        }
        response = service.users().insert(user=user).execute()
        self.vault_client.secrets.kv.create_or_update_secret(
            path=f'gcp_accounts/{response["primaryEmail"]}',
            secret={'email': response['primaryEmail'], 'password': user['password']}
        )
        return response['primaryEmail']

    def create_azure_subscription(self, subscription_name):
        """Create an Azure subscription."""
        client = SubscriptionClient(self.azure_creds)
        response = client.subscriptions.create(
            subscription_name=subscription_name,
            billing_scope=f"/providers/Microsoft.Billing/billingAccounts/{os.getenv('AZURE_BILLING_ACCOUNT')}"
        )
        subscription_id = response.subscription_id
        self.vault_client.secrets.kv.create_or_update_secret(
            path=f'azure_subscriptions/{subscription_id}',
            secret={'subscription_id': subscription_id, 'name': subscription_name}
        )
        return subscription_id