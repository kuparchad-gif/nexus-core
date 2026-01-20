"""
Weaviate Client for Cloud Viren

This module provides a client for interacting with the Weaviate vector database
used by Cloud Viren for technical knowledge, problem-solving concepts, and
troubleshooting tools.
"""

import weaviate
import os
import json
from typing import Dict, Any, List, Optional
from weaviate.util import generate_uuid5

class CloudVirenWeaviateClient:
    def __init__(self, url: str = "http://localhost:8080", api_key: Optional[str] = None):
        """
        Initialize the Weaviate client for Cloud Viren.
        
        Args:
            url: URL of the Weaviate instance
            api_key: API key for authentication (if required)
        """
        auth_config = weaviate.auth.AuthApiKey(api_key=api_key) if api_key else None
        
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-Cloud-Viren-Client": "weaviate-python/1.0.0"
            }
        )
        
    def add_technical_knowledge(self, title: str, content: str, category: str, 
                               tags: List[str], source: str) -> str:
        """
        Add technical knowledge to the Weaviate database.
        
        Args:
            title: Title of the knowledge article
            content: Main content of the knowledge article
            category: Category of the knowledge
            tags: Tags associated with the knowledge
            source: Source of the knowledge
            
        Returns:
            UUID of the created object
        """
        data_object = {
            "title": title,
            "content": content,
            "category": category,
            "tags": tags,
            "source": source,
            "lastUpdated": self._get_current_date()
        }
        
        uuid = generate_uuid5(title)
        
        self.client.data_object.create(
            data_object=data_object,
            class_name="TechnicalKnowledge",
            uuid=uuid
        )
        
        return uuid
        
    def add_problem_solving_concept(self, name: str, description: str, 
                                   applicability: List[str], steps: List[str],
                                   examples: List[str], related_concepts: List[str]) -> str:
        """
        Add a problem-solving concept to the Weaviate database.
        
        Args:
            name: Name of the problem-solving concept
            description: Description of the concept
            applicability: Areas where this concept is applicable
            steps: Steps involved in applying this concept
            examples: Examples of the concept in action
            related_concepts: Related problem-solving concepts
            
        Returns:
            UUID of the created object
        """
        data_object = {
            "name": name,
            "description": description,
            "applicability": applicability,
            "steps": steps,
            "examples": examples,
            "relatedConcepts": related_concepts
        }
        
        uuid = generate_uuid5(name)
        
        self.client.data_object.create(
            data_object=data_object,
            class_name="ProblemSolvingConcept",
            uuid=uuid
        )
        
        return uuid
        
    def add_troubleshooting_tool(self, name: str, description: str, usage: str,
                                parameters: List[str], output_format: str,
                                category: str, compatible_systems: List[str]) -> str:
        """
        Add a troubleshooting tool to the Weaviate database.
        
        Args:
            name: Name of the troubleshooting tool
            description: Description of the tool
            usage: How to use the tool
            parameters: Parameters accepted by the tool
            output_format: Format of the tool's output
            category: Category of the tool
            compatible_systems: Systems compatible with this tool
            
        Returns:
            UUID of the created object
        """
        data_object = {
            "name": name,
            "description": description,
            "usage": usage,
            "parameters": parameters,
            "outputFormat": output_format,
            "category": category,
            "compatibleSystems": compatible_systems
        }
        
        uuid = generate_uuid5(name)
        
        self.client.data_object.create(
            data_object=data_object,
            class_name="TroubleshootingTool",
            uuid=uuid
        )
        
        return uuid
        
    def search_technical_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for technical knowledge using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching knowledge articles
        """
        result = (
            self.client.query
            .get("TechnicalKnowledge", ["title", "content", "category", "tags", "source", "lastUpdated"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        
        return self._extract_results(result, "TechnicalKnowledge")
        
    def search_problem_solving_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for problem-solving concepts using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching problem-solving concepts
        """
        result = (
            self.client.query
            .get("ProblemSolvingConcept", ["name", "description", "applicability", "steps", "examples", "relatedConcepts"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        
        return self._extract_results(result, "ProblemSolvingConcept")
        
    def search_troubleshooting_tools(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for troubleshooting tools using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching troubleshooting tools
        """
        result = (
            self.client.query
            .get("TroubleshootingTool", ["name", "description", "usage", "parameters", "outputFormat", "category", "compatibleSystems"])
            .with_near_text({"concepts": [query]})
            .with_limit(limit)
            .do()
        )
        
        return self._extract_results(result, "TroubleshootingTool")
        
    def create_cross_reference(self, from_class: str, from_uuid: str, 
                              to_class: str, to_uuid: str, reference_property: str) -> None:
        """
        Create a cross-reference between two objects.
        
        Args:
            from_class: Class of the source object
            from_uuid: UUID of the source object
            to_class: Class of the target object
            to_uuid: UUID of the target object
            reference_property: Name of the reference property
        """
        self.client.data_object.reference.add(
            from_class_name=from_class,
            from_uuid=from_uuid,
            from_property_name=reference_property,
            to_class_name=to_class,
            to_uuid=to_uuid
        )
        
    def _extract_results(self, result: Dict[str, Any], class_name: str) -> List[Dict[str, Any]]:
        """
        Extract results from a Weaviate query response.
        
        Args:
            result: Query response from Weaviate
            class_name: Name of the class being queried
            
        Returns:
            List of result objects
        """
        if "data" not in result or "Get" not in result["data"] or class_name not in result["data"]["Get"]:
            return []
            
        return result["data"]["Get"][class_name]
        
    def _get_current_date(self) -> str:
        """
        Get the current date in ISO format.
        
        Returns:
            Current date in ISO format
        """
        from datetime import datetime
        return datetime.now().isoformat()