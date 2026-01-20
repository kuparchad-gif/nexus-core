from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
from langchain.agents import AgentExecutor, create_react_agent
from qdrant_client import QdrantClient, PointStruct
from transformers import AutoModel, AutoTokenizer
from vault_carrier import VaultCarrier
import os
import uuid
import hashlib

class SQLiteFarm:
    def __init__(self, farm_id, db_path=f'/app/sqlite/farm_{uuid.uuid4()}.db'):
        self.farm_id = farm_id
        self.conn = sqlite3.connect(db_path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS designs (id TEXT PRIMARY KEY, type TEXT, code TEXT, prompt TEXT)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_id ON designs(id)')
        self.vault = VaultCarrier()
        self.qdrant = QdrantClient(
            url=self.vault.retrieve_credentials('qdrant')['url'],
            api_key=self.vault.retrieve_credentials('qdrant')['api_key']
        )
        self.llm = AutoModel.from_pretrained('meta-llama/Llama-3.1-1B', load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-1B')
        self.vectorizer = TfidfVectorizer()
        self.update_tfidf_index()
        from langchain.tools import Tool
        self.tools = [
            Tool(name="SQLiteQuery", func=self.query_db, description="Query SQLite for records"),
            Tool(name="QdrantSearch", func=self.search_qdrant, description="Search Qdrant for embeddings"),
            Tool(name="TFIDFSearch", func=self.search_tfidf, description="Fast local text search")
        ]
        self.agent = create_react_agent(self.llm, self.tools)

    def query_db(self, query_str):
        conditions = query_str.split(' AND ')
        query = 'SELECT * FROM designs WHERE ' + ' AND '.join([f"{c.split('==')[0].strip()} = ?" for c in conditions])
        params = [c.split('==')[1].strip() for c in conditions]
        return self.conn.execute(query, params).fetchall()

    def search_qdrant(self, query):
        vector = self.tokenizer(query, return_tensors='pt').mean(dim=1).detach().numpy()
        results = self.qdrant.search(collection_name='designs', query_vector=vector, limit=3)
        return [hit.payload for hit in results]

    def update_tfidf_index(self):
        records = self.conn.execute('SELECT prompt FROM designs').fetchall()
        prompts = [r[0] for r in records]
        self.tfidf_matrix = self.vectorizer.fit_transform(prompts) if prompts else None

    def search_tfidf(self, query):
        if not self.tfidf_matrix:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix * query_vec.T).toarray()
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
        return [self.conn.execute('SELECT * FROM designs LIMIT 1 OFFSET ?', (i,)).fetchone() for i, _ in results]

    def write_record(self, table, record):
        record_id = hashlib.md5(str(record).encode()).hexdigest()
        self.conn.execute(f'INSERT INTO {table} (id, type, code, prompt) VALUES (?, ?, ?, ?)', 
                         (record_id, record.get('type'), record.get('code'), record.get('prompt')))
        self.conn.commit()
        self.update_tfidf_index()
        vector = self.tokenizer(str(record), return_tensors='pt').mean(dim=1).detach().numpy()
        self.qdrant.upsert(
            collection_name='designs',
            points=[PointStruct(id=record_id, vector=vector, payload=record)]
        )
        return record_id