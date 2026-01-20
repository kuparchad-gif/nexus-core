from google.cloud import firestore
from config import FLUX_TOKEN

class StorageDriver:
    def write(self, key, data):
        raise NotImplementedError

    def read(self, key):
        raise NotImplementedError

    def search(self, query):
        raise NotImplementedError

class FirestoreMemoryDriver(StorageDriver):
    def __init__(self):
        self.db = firestore.Client()
        self.collection = self.db.collection("nova_memory")

    def write(self, key, data):
        doc_ref = self.collection.document(key)
        doc_ref.set(data)
        return {"status": "success", "action": "write", "key": key}

    def read(self, key):
        doc = self.collection.document(key).get()
        if doc.exists:
            return doc.to_dict()
        return {"error": "Not found", "key": key}

    def search(self, query):
        results = self.collection.where("content", "==", query).stream()
        return [doc.to_dict() for doc in results]

class MemoryRouter:
    def __init__(self):
        self.primary = FirestoreMemoryDriver()
        self.secondary = None  # Placeholder for fallback system like IPFS or Supabase

    def triage(self, operation, key, data=None):
        if operation == "write":
            return self.primary.write(key, data)
        elif operation == "read":
            return self.primary.read(key)
        elif operation == "search":
            return self.primary.search(key)
        else:
            return {"error": "Unknown operation"}
