# File: root/scripts/session_manager.py
# Purpose: Persistent session storage, loading, updating, and DB redundancy for Engineers

import uuid
from datetime import datetime

try:
    import weaviate
    from weaviate.client import WeaviateClient
    from weaviate.connect import ConnectionParams
    from weaviate.classes.config import Property, DataType

    # Use the new v4 client
    connection_params = ConnectionParams.from_url("http://localhost:8080")
    client = WeaviateClient(connection_params)
except ImportError:
    print("Weaviate client not found. Using fallback mode.")
    client = None
except Exception as e:
    print(f"Error connecting to Weaviate: {e}")
    client = None

SESSION_CLASS = "Session"
METADATA_CLASS = "SessionMetadata"
MODULE_CLASS = "ModuleBlueprint"


def ensure_schema():
    if client is None:
        return
        
    try:
        # Get schema in v4 client
        schema = client.schema.get()
        existing_classes = {cls.name for cls in schema.classes}

        if SESSION_CLASS not in existing_classes:
            # Create class in v4 client
            properties = [
                Property(name="session_id", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.DATE)
            ]
            
            client.collections.create(
                name=SESSION_CLASS,
                properties=properties
            )

        if METADATA_CLASS not in existing_classes:
            # Create metadata class in v4 client
            properties = [
                Property(name="last_saved_session", data_type=DataType.TEXT),
                Property(name="last_loaded_session", data_type=DataType.TEXT)
            ]
            
            client.collections.create(
                name=METADATA_CLASS,
                properties=properties
            )

        if MODULE_CLASS not in existing_classes:
            # Create module class in v4 client
            properties = [
                Property(name="module_name", data_type=DataType.TEXT),
                Property(name="version", data_type=DataType.TEXT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="last_modified", data_type=DataType.DATE),
                Property(name="dependencies", data_type=DataType.TEXT_ARRAY),
                Property(name="status", data_type=DataType.TEXT)
            ]
            
            client.collections.create(
                name=MODULE_CLASS,
                properties=properties
            )
    except Exception as e:
        print(f"Error ensuring schema: {e}")


def save_session(session_id, content):
    if client is None:
        print("Weaviate client not available. Session not saved.")
        return
        
    try:
        properties = {
            "session_id": session_id,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        # Create object in v4 client
        collection = client.collections.get(SESSION_CLASS)
        collection.data.insert(properties)
        update_session_metadata(session_id)
    except Exception as e:
        print(f"Error saving session: {e}")


def load_sessions():
    if client is None:
        print("Weaviate client not available. Returning empty sessions list.")
        return []
        
    try:
        # Query in v4 client
        collection = client.collections.get(SESSION_CLASS)
        result = collection.query.fetch_objects()
        return [{"session_id": obj.properties.get("session_id"), 
                 "content": obj.properties.get("content"), 
                 "timestamp": obj.properties.get("timestamp")} 
                for obj in result.objects]
    except Exception as e:
        print(f"Error loading sessions: {e}")
        return []


def append_to_latest_session(new_content):
    if client is None:
        print("Weaviate client not available. Session not appended.")
        return
        
    try:
        metadata = get_session_metadata()
        session_id = metadata.get("last_saved_session")

        if not session_id:
            session_id = str(uuid.uuid4())
            save_session(session_id, new_content)
            return

        # Find object by session_id in v4 client
        collection = client.collections.get(SESSION_CLASS)
        results = collection.query.fetch_objects(
            filters=collection.query.filter.by_property("session_id").equal(session_id)
        )

        if not results.objects:
            save_session(session_id, new_content)
            return

        obj = results.objects[0]
        updated_content = obj.properties.get("content", "") + "\n" + new_content

        # Update object in v4 client
        collection.data.update(
            uuid=obj.uuid,
            properties={"content": updated_content}
        )
    except Exception as e:
        print(f"Error appending to session: {e}")


def get_session_metadata():
    if client is None:
        return {}
        
    try:
        # Query in v4 client
        collection = client.collections.get(METADATA_CLASS)
        result = collection.query.fetch_objects()
        
        if not result.objects:
            return {}
            
        obj = result.objects[0]
        return {
            "last_saved_session": obj.properties.get("last_saved_session"),
            "last_loaded_session": obj.properties.get("last_loaded_session"),
            "uuid": obj.uuid
        }
    except Exception as e:
        print(f"Error getting session metadata: {e}")
        return {}


def update_session_metadata(session_id):
    if client is None:
        return
        
    try:
        data = get_session_metadata()
        collection = client.collections.get(METADATA_CLASS)
        
        if data and "uuid" in data:
            # Update in v4 client
            collection.data.update(
                uuid=data["uuid"],
                properties={"last_saved_session": session_id}
            )
        else:
            # Create in v4 client
            collection.data.insert({
                "last_saved_session": session_id
            })
    except Exception as e:
        print(f"Error updating session metadata: {e}")


def get_last_session_id():
    data = get_session_metadata()
    return data.get("last_saved_session")


def register_module(module_name, version, file_path, last_modified, dependencies, status):
    if client is None:
        print("Weaviate client not available. Module not registered.")
        return
        
    try:
        properties = {
            "module_name": module_name,
            "version": version,
            "file_path": file_path,
            "last_modified": last_modified,
            "dependencies": dependencies,
            "status": status
        }
        # Create in v4 client
        collection = client.collections.get(MODULE_CLASS)
        collection.data.insert(properties)
    except Exception as e:
        print(f"Error registering module: {e}")


def list_modules():
    if client is None:
        print("Weaviate client not available. Returning empty modules list.")
        return []
        
    try:
        # Query in v4 client
        collection = client.collections.get(MODULE_CLASS)
        result = collection.query.fetch_objects()
        
        return [{
            "module_name": obj.properties.get("module_name"),
            "version": obj.properties.get("version"),
            "status": obj.properties.get("status"),
            "file_path": obj.properties.get("file_path"),
            "last_modified": obj.properties.get("last_modified")
        } for obj in result.objects]
    except Exception as e:
        print(f"Error listing modules: {e}")
        return []


# Initialize schema if client is available
try:
    ensure_schema()
except Exception as e:
    print(f"Error initializing schema: {e}. Continuing in fallback mode.")