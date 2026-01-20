from qdrant_client import QdrantClient
import boto3

class LLMRegistry:
    def __init__(self, regions=['us-east-1', 'eu-west-1']):
        self.regions = regions
        self.databases = {region: QdrantClient(host=f'db-{region}.localhost', port=6333) for region in regions}
        self.dynamodb = boto3.client('dynamodb')  # For metadata

    def register(self, llm_data):
        llm_id = llm_data['id']
        language = llm_data['language']
        for region, db in self.databases.items():
            db.upload_collection(
                collection_name="llm_registry",
                vectors=[self.encode_llm(llm_data)],
                payload={'id': llm_id, 'language': language, 'region': region}
            )
        self.dynamodb.put_item(
            TableName='LLMMetadata',
            Item={'llm_id': {'S': llm_id}, 'language': {'S': language}}
        )

    def encode_llm(self, llm_data): return [0.1] * 768

    def get_database(self):
        return self.databases
