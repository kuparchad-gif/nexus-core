import os
import requests
import json
import logging
from typing import Dict, List, Any, Optional, Union
import weaviate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'viren_sync.log')
)
logger = logging.getLogger('viren_sync.weaviate_manager')

class WeaviateManager:
    """Manages Weaviate database operations using TinyLlama model."""
    
    def __init__(
        self, 
        weaviate_url: str,
        model_endpoint: Optional[str] = None,
        local_model_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.weaviate_url = weaviate_url
        self.model_endpoint = model_endpoint or os.environ.get("TINYLLAMA_MODEL_ENDPOINT")
        self.local_model_path = local_model_path
        self.api_key = api_key
        
        # Initialize local model if available
        self.local_model = self._init_local_model() if local_model_path else None
        
        # Initialize Weaviate client
        self.client = self._init_weaviate_client()
        logger.info(f"WeaviateManager initialized with URL: {weaviate_url}")
    
    def _init_local_model(self):
        """Initialize local TinyLlama model if available."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading local TinyLlama model from {self.local_model_path}")
            
            model_id = self.local_model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            logger.info("Local TinyLlama model loaded successfully")
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"Error initializing local TinyLlama model: {str(e)}")
            return None
    
    def _init_weaviate_client(self):
        """Initialize Weaviate client."""
        try:
            auth_config = None
            if self.api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key=self.api_key)
                
            client = weaviate.Client(url=self.weaviate_url, auth_client_secret=auth_config)
            logger.info(f"Weaviate client connected to {self.weaviate_url}")
            return client
        except Exception as e:
            logger.error(f"Error initializing Weaviate client: {str(e)}")
            raise
    
    def get_schema(self) -> Dict:
        """Get Weaviate schema."""
        try:
            schema = self.client.schema.get()
            logger.info(f"Retrieved schema with {len(schema.get('classes', []))} classes")
            return schema
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}")
            return {"classes": []}
    
    def generate_query(self, description: str) -> Dict:
        """Generate Weaviate query from natural language description."""
        schema = json.dumps(self.get_schema())
        
        prompt = f"""Weaviate Schema:
{schema}

Task: {description}

Generate a Weaviate GraphQL query to accomplish this task. Return only the query without any explanation."""
        
        logger.info(f"Generating query for: {description}")
        
        if self.local_model:
            query = self._generate_local(prompt)
        elif self.model_endpoint:
            query = self._generate_remote(prompt)
        else:
            logger.error("No TinyLlama model available (neither local nor remote)")
            raise ValueError("No TinyLlama model available (neither local nor remote)")
        
        logger.info(f"Generated query: {query[:100]}...")
        return json.loads(query) if query.strip().startswith("{") else {"query": query}
    
    def _generate_local(self, prompt: str) -> str:
        """Generate query using local model."""
        import torch
        
        logger.info("Generating query using local model")
        
        model = self.local_model["model"]
        tokenizer = self.local_model["tokenizer"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the query
        query = response.split(prompt)[-1].strip()
        return query
    
    def _generate_remote(self, prompt: str) -> str:
        """Generate query using remote endpoint."""
        logger.info(f"Generating query using remote endpoint: {self.model_endpoint}")
        
        try:
            response = requests.post(
                self.model_endpoint,
                json={"prompt": prompt, "max_tokens": 512, "temperature": 0.2}
            )
            
            if response.status_code != 200:
                logger.error(f"TinyLlama API error: {response.status_code} - {response.text}")
                raise Exception(f"TinyLlama API error: {response.status_code} - {response.text}")
            
            return response.json()["output"].strip()
        except Exception as e:
            logger.error(f"Error calling remote TinyLlama API: {str(e)}")
            raise
    
    def execute_query(self, query: Dict) -> Dict:
        """Execute a GraphQL query against Weaviate."""
        try:
            if "query" in query:
                # Simple string query
                logger.info(f"Executing string query: {query['query'][:100]}...")
                return self.client.query.raw(query["query"])
            else:
                # Complex query object
                logger.info(f"Executing complex query: {json.dumps(query)[:100]}...")
                return self.client.query.raw(json.dumps(query))
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def execute_natural_language(self, description: str) -> Dict:
        """Generate and execute a query from natural language."""
        logger.info(f"Processing natural language query: {description}")
        
        try:
            query = self.generate_query(description)
            results = self.execute_query(query)
            logger.info("Query executed successfully")
            return {
                "query": query,
                "results": results,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error in natural language query: {str(e)}")
            return {
                "query": None,
                "results": None,
                "success": False,
                "error": str(e)
            }
    
    def create_class(self, class_name: str, properties: List[Dict], description: str = "") -> bool:
        """Create a new class in Weaviate."""
        logger.info(f"Creating class: {class_name}")
        
        class_obj = {
            "class": class_name,
            "description": description,
            "properties": properties
        }
        
        try:
            self.client.schema.create_class(class_obj)
            logger.info(f"Class {class_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating class {class_name}: {str(e)}")
            return False
    
    def add_data(self, class_name: str, data_objects: List[Dict], batch_size: int = 100) -> int:
        """Add data objects to a Weaviate class."""
        logger.info(f"Adding {len(data_objects)} objects to class {class_name}")
        
        try:
            with self.client.batch as batch:
                batch.batch_size = batch_size
                
                for data_object in data_objects:
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=class_name
                    )
            
            logger.info(f"Added {len(data_objects)} objects to {class_name}")
            return len(data_objects)
        except Exception as e:
            logger.error(f"Error adding data to {class_name}: {str(e)}")
            raise
    
    def natural_language_schema_update(self, description: str) -> Dict:
        """Update schema based on natural language description."""
        logger.info(f"Processing schema update: {description}")
        
        prompt = f"""Current Weaviate Schema:
{json.dumps(self.get_schema())}

Task: {description}

Generate a Weaviate class definition to accomplish this task. Return a JSON object with 'class_name', 'description', and 'properties' fields."""
        
        try:
            if self.local_model:
                response = self._generate_local(prompt)
            elif self.model_endpoint:
                response = self._generate_remote(prompt)
            else:
                logger.error("No TinyLlama model available")
                raise ValueError("No TinyLlama model available")
            
            schema_update = json.loads(response)
            
            # Create the class
            success = self.create_class(
                class_name=schema_update["class_name"],
                properties=schema_update["properties"],
                description=schema_update.get("description", "")
            )
            
            logger.info(f"Schema update {'successful' if success else 'failed'}")
            return {
                "schema_update": schema_update,
                "success": success
            }
        except Exception as e:
            logger.error(f"Error in schema update: {str(e)}")
            return {
                "schema_update": None,
                "success": False,
                "error": str(e),
                "raw_response": response if 'response' in locals() else None
            }

# Factory function to create Weaviate managers for different locations
def create_weaviate_manager(location: str) -> WeaviateManager:
    """Create a Weaviate manager for a specific location."""
    if location.lower() == "cloud":
        # Cloud Weaviate
        url = os.environ.get("CLOUD_WEAVIATE_URL", "https://your-modal-weaviate-url")
        api_key = os.environ.get("CLOUD_WEAVIATE_API_KEY")
        model_endpoint = os.environ.get("CLOUD_TINYLLAMA_ENDPOINT")
        logger.info(f"Creating cloud Weaviate manager with URL: {url}")
        return WeaviateManager(weaviate_url=url, model_endpoint=model_endpoint, api_key=api_key)
    
    elif location.lower() in ["local", "desktop"]:
        # Local Weaviate for Desktop
        url = os.environ.get("LOCAL_WEAVIATE_URL", "http://localhost:8080")
        model_path = os.environ.get("LOCAL_TINYLLAMA_PATH", "models/tinyllama-coder-en-v0.1")
        logger.info(f"Creating desktop Weaviate manager with URL: {url}")
        return WeaviateManager(weaviate_url=url, local_model_path=model_path)
    
    elif location.lower() == "lillith":
        # Local Weaviate for Lillith
        url = os.environ.get("LILLITH_WEAVIATE_URL", "http://localhost:8081")
        model_path = os.environ.get("LOCAL_TINYLLAMA_PATH", "models/tinyllama-coder-en-v0.1")
        logger.info(f"Creating Lillith Weaviate manager with URL: {url}")
        return WeaviateManager(weaviate_url=url, local_model_path=model_path)
    
    else:
        logger.error(f"Unknown location: {location}")
        raise ValueError(f"Unknown location: {location}")

if __name__ == "__main__":
    # Example usage
    manager = create_weaviate_manager("local")
    
    # Create a class
    manager.create_class(
        class_name="Article",
        properties=[
            {
                "name": "title",
                "dataType": ["text"]
            },
            {
                "name": "content",
                "dataType": ["text"]
            },
            {
                "name": "url",
                "dataType": ["string"]
            }
        ],
        description="News articles"
    )
    
    # Query using natural language
    result = manager.execute_natural_language("Find all articles about technology")
    print(result)