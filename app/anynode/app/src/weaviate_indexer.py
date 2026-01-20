"""
Weaviate Indexer for Cloud Viren

This module provides functionality for indexing technical content into the
Weaviate vector database used by Cloud Viren.
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Generator
from pathlib import Path

from .weaviate_client import CloudVirenWeaviateClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateIndexer:
    def __init__(self, client: CloudVirenWeaviateClient):
        """
        Initialize the Weaviate indexer.
        
        Args:
            client: CloudVirenWeaviateClient instance
        """
        self.client = client
        
    def index_technical_documentation(self, docs_dir: str, 
                                     source: str = "internal") -> Tuple[int, int, int]:
        """
        Index technical documentation from a directory.
        
        Args:
            docs_dir: Directory containing technical documentation
            source: Source identifier for the documentation
            
        Returns:
            Tuple of (processed_count, success_count, error_count)
        """
        processed = 0
        success = 0
        errors = 0
        
        logger.info(f"Indexing technical documentation from {docs_dir}")
        
        for file_path, content, metadata in self._read_documentation_files(docs_dir):
            processed += 1
            
            try:
                # Extract title from filename or metadata
                title = metadata.get("title", Path(file_path).stem.replace("_", " ").title())
                
                # Extract category from directory structure or metadata
                category = metadata.get("category", self._extract_category_from_path(file_path))
                
                # Extract tags from metadata or content
                tags = metadata.get("tags", self._extract_tags_from_content(content))
                
                # Add to Weaviate
                self.client.add_technical_knowledge(
                    title=title,
                    content=content,
                    category=category,
                    tags=tags,
                    source=source
                )
                
                success += 1
                logger.info(f"Indexed: {title}")
                
            except Exception as e:
                errors += 1
                logger.error(f"Error indexing {file_path}: {str(e)}")
                
        logger.info(f"Indexing complete. Processed: {processed}, Success: {success}, Errors: {errors}")
        return processed, success, errors
        
    def index_problem_solving_concepts(self, concepts_file: str) -> Tuple[int, int, int]:
        """
        Index problem-solving concepts from a JSON file.
        
        Args:
            concepts_file: Path to JSON file containing problem-solving concepts
            
        Returns:
            Tuple of (processed_count, success_count, error_count)
        """
        processed = 0
        success = 0
        errors = 0
        
        logger.info(f"Indexing problem-solving concepts from {concepts_file}")
        
        try:
            with open(concepts_file, 'r', encoding='utf-8') as f:
                concepts = json.load(f)
                
            for concept in concepts:
                processed += 1
                
                try:
                    self.client.add_problem_solving_concept(
                        name=concept["name"],
                        description=concept["description"],
                        applicability=concept.get("applicability", []),
                        steps=concept.get("steps", []),
                        examples=concept.get("examples", []),
                        related_concepts=concept.get("relatedConcepts", [])
                    )
                    
                    success += 1
                    logger.info(f"Indexed concept: {concept['name']}")
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Error indexing concept {concept.get('name', 'unknown')}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error reading concepts file {concepts_file}: {str(e)}")
            
        logger.info(f"Indexing complete. Processed: {processed}, Success: {success}, Errors: {errors}")
        return processed, success, errors
        
    def index_troubleshooting_tools(self, tools_file: str) -> Tuple[int, int, int]:
        """
        Index troubleshooting tools from a JSON file.
        
        Args:
            tools_file: Path to JSON file containing troubleshooting tools
            
        Returns:
            Tuple of (processed_count, success_count, error_count)
        """
        processed = 0
        success = 0
        errors = 0
        
        logger.info(f"Indexing troubleshooting tools from {tools_file}")
        
        try:
            with open(tools_file, 'r', encoding='utf-8') as f:
                tools = json.load(f)
                
            for tool in tools:
                processed += 1
                
                try:
                    self.client.add_troubleshooting_tool(
                        name=tool["name"],
                        description=tool["description"],
                        usage=tool.get("usage", ""),
                        parameters=tool.get("parameters", []),
                        output_format=tool.get("outputFormat", ""),
                        category=tool.get("category", "general"),
                        compatible_systems=tool.get("compatibleSystems", [])
                    )
                    
                    success += 1
                    logger.info(f"Indexed tool: {tool['name']}")
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Error indexing tool {tool.get('name', 'unknown')}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error reading tools file {tools_file}: {str(e)}")
            
        logger.info(f"Indexing complete. Processed: {processed}, Success: {success}, Errors: {errors}")
        return processed, success, errors
        
    def create_cross_references(self) -> int:
        """
        Create cross-references between related objects in the database.
        
        Returns:
            Number of cross-references created
        """
        # This is a placeholder for a more sophisticated implementation
        # In a real implementation, this would analyze the content and create
        # appropriate cross-references based on semantic similarity
        
        logger.info("Creating cross-references between objects")
        return 0
        
    def _read_documentation_files(self, docs_dir: str) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        """
        Read documentation files from a directory.
        
        Args:
            docs_dir: Directory containing documentation files
            
        Yields:
            Tuples of (file_path, content, metadata)
        """
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(('.md', '.txt', '.html')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Extract metadata from content (e.g., YAML frontmatter in Markdown)
                        metadata = self._extract_metadata(content, file_path)
                        
                        yield file_path, content, metadata
                        
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
                        
    def _extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from content.
        
        Args:
            content: File content
            file_path: Path to the file
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Check for YAML frontmatter in Markdown files
        if file_path.endswith('.md'):
            frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            if frontmatter_match:
                try:
                    import yaml
                    frontmatter = frontmatter_match.group(1)
                    metadata = yaml.safe_load(frontmatter)
                except Exception as e:
                    logger.warning(f"Error parsing frontmatter in {file_path}: {str(e)}")
                    
        return metadata
        
    def _extract_category_from_path(self, file_path: str) -> str:
        """
        Extract category from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Category string
        """
        parts = Path(file_path).parts
        if len(parts) > 1:
            return parts[-2].replace("_", " ").title()
        return "General"
        
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """
        Extract tags from content.
        
        Args:
            content: File content
            
        Returns:
            List of tags
        """
        # Simple tag extraction from content
        # In a real implementation, this would use NLP techniques
        tags = set()
        
        # Look for hashtags
        hashtags = re.findall(r'#(\w+)', content)
        tags.update(hashtags)
        
        # Look for "Tags:" or "Keywords:" sections
        tag_section = re.search(r'(?:Tags|Keywords):\s*(.*?)(?:\n\n|\Z)', content, re.IGNORECASE)
        if tag_section:
            section_tags = [t.strip() for t in tag_section.group(1).split(',')]
            tags.update(section_tags)
            
        return list(tags)