import boto3
from google.cloud import firestore
from azure.cosmos import CosmosClient
import hvac
import os
import uuid
from datetime import datetime

class MasterDBManager:
    def __init__(self):
        self.vault_client = hvac.Client(url=os.getenv('VAULT_URL'), token=os.getenv('VAULT_TOKEN'))
        self.aws_creds = self.vault_client.secrets.kv.read_secret_version(path='aws')['data']['data']
        self.gcp_creds_path = 'gcp_credentials.json'
        self.azure_creds = self.vault_client.secrets.kv.read_secret_version(path='azure')['data']['data']
        self.dynamodb = boto3.resource('dynamodb', **self.aws_creds)
        self.firestore = firestore.Client.from_service_account_json(self.gcp_creds_path)
        self.cosmos = CosmosClient(
            self.azure_creds['cosmos_endpoint'],
            self.azure_creds['cosmos_key']
        ).get_database_client('lillith_db')
        self.services = [
            'cloud_accounts', 'shopify', 'etsy', 'pionex', 'social_media', 'viren_tech', 'codex'
        ]

    def initialize_databases(self):
        """Create tables/collections for each service in all clouds."""
        for service in self.services:
            # AWS DynamoDB
            try:
                self.dynamodb.create_table(
                    TableName=service,
                    KeySchema=[{'AttributeName': 'id', 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': 'id', 'AttributeType': 'S'}],
                    BillingMode='PAY_PER_REQUEST'
                )
            except self.dynamodb.meta.client.exceptions.ResourceInUseException:
                pass

            # GCP Firestore
            self.firestore.collection(service)

            # Azure Cosmos DB
            cosmos_db = self.cosmos.create_container_if_not_exists(
                id=service,
                partition_key={'paths': ['/id'], 'kind': 'Hash'}
            )

    def write_record(self, service, data):
        """Write a record to all three clouds with RAID-like mirroring."""
        data['id'] = str(uuid.uuid4())
        data['timestamp'] = datetime.utcnow().isoformat()
        
        # AWS DynamoDB
        table = self.dynamodb.Table(service)
        table.put_item(Item=data)

        # GCP Firestore
        doc_ref = self.firestore.collection(service).document(data['id'])
        doc_ref.set(data)

        # Azure Cosmos DB
        container = self.cosmos.get_container_client(service)
        container.upsert_item(data)

        self.vault_client.secrets.kv.create_or_update_secret(
            path=f'db_records/{service}/{data["id"]}',
            secret={'data': data, 'clouds': ['aws', 'gcp', 'azure']}
        )
        return data['id']

    def read_record(self, service, record_id):
        """Read a record from primary (AWS) with fallback to mirrors."""
        try:
            table = self.dynamodb.Table(service)
            response = table.get_item(Key={'id': record_id})
            if 'Item' in response:
                return response['Item']
        except Exception as e:
            print(f"AWS read failed: {e}")

        try:
            doc_ref = self.firestore.collection(service).document(record_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            print(f"GCP read failed: {e}")

        try:
            container = self.cosmos.get_container_client(service)
            item = container.read_item(item=record_id, partition_key=record_id)
            return item
        except Exception as e:
            print(f"Azure read failed: {e}")

        raise Exception("Record not found in any cloud")

    def sync_databases(self, service):
        """Ensure consistency across clouds by syncing missing records."""
        aws_records = {item['id']: item for item in self.dynamodb.Table(service).scan()['Items']}
        gcp_records = {doc.id: doc.to_dict() for doc in self.firestore.collection(service).stream()}
        azure_records = {item['id']: item for item in self.cosmos.get_container_client(service).read_all_items()}

        all_ids = set(aws_records.keys()) | set(gcp_records.keys()) | set(azure_records.keys())
        for record_id in all_ids:
            record = aws_records.get(record_id) or gcp_records.get(record_id) or azure_records.get(record_id)
            if record_id not in aws_records:
                self.dynamodb.Table(service).put_item(Item=record)
            if record_id not in gcp_records:
                self.firestore.collection(service).document(record_id).set(record)
            if record_id not in azure_records:
                self.cosmos.get_container_client(service).upsert_item(record)