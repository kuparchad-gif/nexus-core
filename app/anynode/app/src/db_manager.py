import os
import sqlite3
import requests
from typing import List, Dict, Any, Optional, Union

class DatabaseManager:
    """Manages database operations using TinyLlama SQL model."""
    
    def __init__(
        self, 
        db_path: str,
        model_endpoint: Optional[str] = None,
        local_model_path: Optional[str] = None
    ):
        self.db_path = db_path
        self.model_endpoint = model_endpoint or os.environ.get("SQL_MODEL_ENDPOINT")
        self.local_model_path = local_model_path
        
        # Initialize local model if available
        self.local_model = self._init_local_model() if local_model_path else None
        
        # Connect to database
        self._ensure_db_exists()
    
    def _init_local_model(self):
        """Initialize local TinyLlama SQL model if available."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_id = self.local_model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            print(f"Error initializing local SQL model: {str(e)}")
            return None
    
    def _ensure_db_exists(self):
        """Create database if it doesn't exist."""
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path))
        
        # Create connection (will create file if it doesn't exist)
        conn = sqlite3.connect(self.db_path)
        conn.close()
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            
            # Check if this is a SELECT query
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
            else:
                conn.commit()
                result = [{"affected_rows": cursor.rowcount}]
                
            return result
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_schema(self) -> str:
        """Get database schema as a string."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                col_defs = [f"{col[1]} {col[2]}" for col in columns]
                schema.append(f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);")
            
            return "\n\n".join(schema)
        finally:
            conn.close()
    
    def generate_query(self, description: str) -> str:
        """Generate SQL query from natural language description."""
        schema = self.get_schema()
        
        prompt = f"""Database Schema:
{schema}

Task: {description}

Generate a SQL query to accomplish this task. Return only the SQL query without any explanation."""
        
        if self.local_model:
            return self._generate_local(prompt)
        elif self.model_endpoint:
            return self._generate_remote(prompt)
        else:
            raise ValueError("No SQL model available (neither local nor remote)")
    
    def _generate_local(self, prompt: str) -> str:
        """Generate SQL using local model."""
        import torch
        
        model = self.local_model["model"]
        tokenizer = self.local_model["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for more deterministic SQL
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the SQL query
        sql = response.split(prompt)[-1].strip()
        return sql
    
    def _generate_remote(self, prompt: str) -> str:
        """Generate SQL using remote endpoint."""
        response = requests.post(
            self.model_endpoint,
            json={"prompt": prompt, "max_tokens": 256, "temperature": 0.1}
        )
        
        if response.status_code != 200:
            raise Exception(f"SQL model API error: {response.status_code} - {response.text}")
        
        return response.json()["output"].strip()
    
    def execute_natural_language(self, description: str) -> Dict[str, Any]:
        """Generate and execute a query from natural language."""
        try:
            query = self.generate_query(description)
            results = self.execute_query(query)
            return {
                "query": query,
                "results": results,
                "success": True
            }
        except Exception as e:
            return {
                "query": None,
                "results": None,
                "success": False,
                "error": str(e)
            }

# Factory function to create database managers for different locations
def create_db_manager(location: str, db_name: str) -> DatabaseManager:
    """Create a database manager for a specific location."""
    if location.lower() == "cloud":
        # Cloud database
        endpoint = os.environ.get("CLOUD_SQL_MODEL_ENDPOINT")
        db_path = f"/tmp/{db_name}.db"  # Temporary path for cloud DB
        return DatabaseManager(db_path=db_path, model_endpoint=endpoint)
    
    elif location.lower() in ["local", "desktop", "lillith"]:
        # Local database
        local_path = os.path.join(os.path.expanduser("~"), ".viren", "databases", f"{db_name}.db")
        model_path = os.environ.get("LOCAL_SQL_MODEL_PATH", "models/tinyllama-coder-sql-en-v0.1")
        return DatabaseManager(db_path=local_path, local_model_path=model_path)
    
    else:
        raise ValueError(f"Unknown location: {location}")

if __name__ == "__main__":
    # Example usage
    db = create_db_manager("local", "test")
    
    # Create a table
    db.execute_query("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Insert data
    db.execute_query("INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com')")
    
    # Query using natural language
    result = db.execute_natural_language("Find all users with their emails")
    print(result)