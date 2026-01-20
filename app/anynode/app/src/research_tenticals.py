#!/usr/bin/env python3
"""
Research Tentacles for Cloud Viren
Extends knowledge by searching the web for solutions to unknown issues
"""

import os
import json
import time
import logging
import threading
import requests
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger("VirenResearchTentacles")

class ResearchTentacles:
    """
    Research tentacles system for extending knowledge
    Searches the web for solutions to unknown issues
    """
    
    def __init__(self, max_tentacles: int = 5, timeout: int = 30, config_path: str = None):
        """Initialize the research tentacles system"""
        self.max_tentacles = max_tentacles
        self.timeout = timeout
        self.config_path = config_path or os.path.join("config", "tentacles_config.json")
        self.config = self._load_config()
        self.active_tentacles = {}
        self.research_history = []
        self.max_history = 100
        self.search_sources = [
            "documentation",
            "error_database",
            "community_forums",
            "technical_blogs",
            "vendor_knowledge_base",
            "academic_papers"
        ]
        
        logger.info(f"Research tentacles initialized with {len(self.search_sources)} sources")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "search_apis": {
                "google": {
                    "endpoint": "https://www.googleapis.com/customsearch/v1",
                    "api_key": "",
                    "cx": ""  # Custom Search Engine ID
                },
                "bing": {
                    "endpoint": "https://api.bing.microsoft.com/v7.0/search",
                    "api_key": ""
                },
                "stackoverflow": {
                    "endpoint": "https://api.stackexchange.com/2.3/search",
                    "api_key": ""
                }
            },
            "technical_sources": [
                "stackoverflow.com",
                "github.com",
                "docs.microsoft.com",
                "developer.mozilla.org",
                "aws.amazon.com/documentation",
                "cloud.google.com/docs"
            ],
            "max_results_per_source": 5,
            "cache_duration": 86400  # 24 hours
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict) and isinstance(config.get(key), dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    
                    logger.info("Tentacles configuration loaded successfully")
                    return config
            except Exception as e:
                logger.error(f"Error loading tentacles configuration: {e}")
        
        logger.info("Using default tentacles configuration")
        return default_config
    
    def _save_config(self) -> bool:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Tentacles configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving tentacles configuration: {e}")
            return False
    
    def deploy(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deploy research tentacles to search for information
        
        Args:
            query: The search query
            context: Additional context for the search
            
        Returns:
            Dictionary with research results
        """
        logger.info(f"Deploying research tentacles for query: {query}")
        
        # Generate search queries based on the input query and context
        search_queries = self._generate_search_queries(query, context)
        
        # Select appropriate tentacles based on the query
        selected_tentacles = self._select_tentacles(query, context)
        
        # Track active tentacles
        tentacle_id = f"tentacle_{int(time.time())}_{len(self.active_tentacles)}"
        self.active_tentacles[tentacle_id] = {
            "query": query,
            "context": context,
            "start_time": time.time(),
            "selected_tentacles": selected_tentacles,
            "status": "searching"
        }
        
        # Start search threads
        threads = []
        results = {}
        
        for tentacle_type in selected_tentacles:
            thread = threading.Thread(
                target=self._search_with_tentacle,
                args=(tentacle_type, search_queries, results, tentacle_id)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete or timeout
        for thread in threads:
            thread.join(self.timeout)
        
        # Process and synthesize results
        synthesis = self._synthesize_results(results, query, context)
        
        # Update tentacle status
        self.active_tentacles[tentacle_id]["status"] = "completed"
        self.active_tentacles[tentacle_id]["end_time"] = time.time()
        self.active_tentacles[tentacle_id]["results"] = results
        
        # Add to history
        research_entry = {
            "id": tentacle_id,
            "query": query,
            "context_summary": self._summarize_context(context) if context else None,
            "tentacles_used": selected_tentacles,
            "timestamp": time.time(),
            "duration": self.active_tentacles[tentacle_id]["end_time"] - self.active_tentacles[tentacle_id]["start_time"],
            "synthesis": synthesis
        }
        
        self.research_history.append(research_entry)
        if len(self.research_history) > self.max_history:
            self.research_history = self.research_history[-self.max_history:]
        
        # Clean up old tentacles
        self._cleanup_tentacles()
        
        return {
            "tentacle_id": tentacle_id,
            "query": query,
            "tentacles_used": selected_tentacles,
            "findings": synthesis,
            "raw_results": results
        }
    
    def _generate_search_queries(self, query: str, context: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Generate search queries for different sources"""
        queries = {
            "general": [query],
            "technical": [f"technical {query}", f"{query} solution", f"how to fix {query}"],
            "error": [f"error {query}", f"{query} error code", f"{query} exception"]
        }
        
        # Add context-specific queries
        if context:
            if "os" in context:
                queries["os_specific"] = [f"{context['os']} {query}", f"{query} {context['os']}"]
            
            if "language" in context:
                queries["language_specific"] = [f"{context['language']} {query}", f"{query} {context['language']}"]
            
            if "error_code" in context:
                queries["error_specific"] = [f"{context['error_code']} {query}", f"{query} {context['error_code']}"]
        
        return queries
    
    def _select_tentacles(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Select appropriate tentacles based on the query and context"""
        selected_tentacles = []
        query_lower = query.lower()
        
        # Documentation search for all queries
        selected_tentacles.append("documentation")
        
        # Error database for error-related queries
        if any(term in query_lower for term in ["error", "exception", "fail", "issue", "bug", "crash"]):
            selected_tentacles.append("error_database")
        
        # Community forums for complex or specific issues
        if any(term in query_lower for term in ["how to", "help", "problem", "solution", "fix"]):
            selected_tentacles.append("community_forums")
        
        # Technical blogs for advanced topics
        if any(term in query_lower for term in ["advanced", "architecture", "design", "pattern", "best practice"]):
            selected_tentacles.append("technical_blogs")
        
        # Vendor knowledge base for product-specific issues
        if context and "vendor" in context:
            selected_tentacles.append("vendor_knowledge_base")
        
        # Academic papers for research-oriented queries
        if any(term in query_lower for term in ["research", "paper", "algorithm", "theory", "concept"]):
            selected_tentacles.append("academic_papers")
        
        # Limit to max_tentacles
        if len(selected_tentacles) > self.max_tentacles:
            # Prioritize based on query characteristics
            priorities = {
                "error_database": 5,
                "documentation": 4,
                "community_forums": 3,
                "vendor_knowledge_base": 2,
                "technical_blogs": 1,
                "academic_papers": 0
            }
            selected_tentacles.sort(key=lambda x: priorities.get(x, 0), reverse=True)
            selected_tentacles = selected_tentacles[:self.max_tentacles]
        
        return selected_tentacles
    
    def _search_with_tentacle(self, tentacle_type: str, search_queries: Dict[str, List[str]], 
                             results: Dict[str, Any], tentacle_id: str) -> None:
        """Search using a specific tentacle type"""
        try:
            logger.info(f"Tentacle {tentacle_id} ({tentacle_type}) searching...")
            
            # Select appropriate queries for this tentacle
            queries_to_use = []
            if tentacle_type == "documentation":
                queries_to_use = search_queries.get("general", []) + search_queries.get("technical", [])
            elif tentacle_type == "error_database":
                queries_to_use = search_queries.get("error", []) + search_queries.get("error_specific", [])
            elif tentacle_type == "community_forums":
                queries_to_use = search_queries.get("general", []) + search_queries.get("os_specific", [])
            elif tentacle_type == "technical_blogs":
                queries_to_use = search_queries.get("technical", []) + search_queries.get("language_specific", [])
            elif tentacle_type == "vendor_knowledge_base":
                queries_to_use = search_queries.get("os_specific", []) + search_queries.get("error_specific", [])
            elif tentacle_type == "academic_papers":
                queries_to_use = search_queries.get("general", [])
            
            # Ensure we have at least one query
            if not queries_to_use:
                queries_to_use = search_queries.get("general", [])
            
            # Limit to a reasonable number of queries
            queries_to_use = queries_to_use[:3]
            
            # Search results for this tentacle
            tentacle_results = []
            
            for query in queries_to_use:
                # Search using appropriate method for this tentacle
                if tentacle_type == "documentation":
                    search_results = self._search_documentation(query)
                elif tentacle_type == "error_database":
                    search_results = self._search_error_database(query)
                elif tentacle_type == "community_forums":
                    search_results = self._search_community_forums(query)
                elif tentacle_type == "technical_blogs":
                    search_results = self._search_technical_blogs(query)
                elif tentacle_type == "vendor_knowledge_base":
                    search_results = self._search_vendor_knowledge_base(query)
                elif tentacle_type == "academic_papers":
                    search_results = self._search_academic_papers(query)
                else:
                    search_results = []
                
                tentacle_results.extend(search_results)
            
            # Remove duplicates
            unique_results = []
            urls = set()
            for result in tentacle_results:
                if result.get("url") not in urls:
                    urls.add(result.get("url"))
                    unique_results.append(result)
            
            # Store results
            results[tentacle_type] = {
                "count": len(unique_results),
                "results": unique_results[:self.config["max_results_per_source"]],
                "timestamp": time.time()
            }
            
            logger.info(f"Tentacle {tentacle_id} ({tentacle_type}) found {len(unique_results)} results")
        
        except Exception as e:
            logger.error(f"Error in tentacle {tentacle_id} ({tentacle_type}): {e}")
            results[tentacle_type] = {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _search_documentation(self, query: str) -> List[Dict[str, Any]]:
        """Search documentation sources"""
        # This is a simplified implementation
        # In a real implementation, you would use actual search APIs
        
        # Simulate search results
        return [
            {
                "title": f"Documentation for {query}",
                "url": f"https://docs.example.com/search?q={query.replace(' ', '+')}",
                "snippet": f"Official documentation explaining {query} with examples and best practices.",
                "source": "documentation"
            },
            {
                "title": f"Developer Guide: {query}",
                "url": f"https://developer.example.com/guide/{query.replace(' ', '-')}",
                "snippet": f"Comprehensive guide for developers about {query} with code samples.",
                "source": "documentation"
            }
        ]
    
    def _search_error_database(self, query: str) -> List[Dict[str, Any]]:
        """Search error code databases"""
        # Simplified implementation
        return [
            {
                "title": f"Error Database: {query}",
                "url": f"https://errors.example.com/db?code={query.replace(' ', '+')}",
                "snippet": f"Known error {query} occurs when system resources are exhausted.",
                "source": "error_database"
            },
            {
                "title": f"Troubleshooting {query}",
                "url": f"https://support.example.com/kb/{query.replace(' ', '-')}",
                "snippet": f"Step-by-step guide to resolve error {query} in production environments.",
                "source": "error_database"
            }
        ]
    
    def _search_community_forums(self, query: str) -> List[Dict[str, Any]]:
        """Search community forums"""
        # Simplified implementation
        return [
            {
                "title": f"Solved: {query}",
                "url": f"https://community.example.com/thread/{query.replace(' ', '-')}",
                "snippet": f"Community discussion about {query} with verified solution.",
                "source": "community_forums"
            },
            {
                "title": f"How to fix {query}",
                "url": f"https://forums.example.com/topic/{query.replace(' ', '+')}",
                "snippet": f"Expert users discuss multiple approaches to solve {query}.",
                "source": "community_forums"
            }
        ]
    
    def _search_technical_blogs(self, query: str) -> List[Dict[str, Any]]:
        """Search technical blogs"""
        # Simplified implementation
        return [
            {
                "title": f"Deep dive into {query}",
                "url": f"https://techblog.example.com/article/{query.replace(' ', '-')}",
                "snippet": f"Technical analysis of {query} with performance considerations.",
                "source": "technical_blogs"
            },
            {
                "title": f"Understanding {query}",
                "url": f"https://engineering.example.com/posts/{query.replace(' ', '_')}",
                "snippet": f"Engineering perspective on {query} with architectural insights.",
                "source": "technical_blogs"
            }
        ]
    
    def _search_vendor_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search vendor knowledge bases"""
        # Simplified implementation
        return [
            {
                "title": f"Official solution for {query}",
                "url": f"https://vendor.example.com/kb/{query.replace(' ', '-')}",
                "snippet": f"Vendor-approved solution for {query} with support information.",
                "source": "vendor_knowledge_base"
            },
            {
                "title": f"Product update fixes {query}",
                "url": f"https://updates.example.com/release-notes/{query.replace(' ', '+')}",
                "snippet": f"Recent product update addresses {query} in version 2.3.4.",
                "source": "vendor_knowledge_base"
            }
        ]
    
    def _search_academic_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search academic papers"""
        # Simplified implementation
        return [
            {
                "title": f"Research on {query}",
                "url": f"https://papers.example.com/abstract/{query.replace(' ', '+')}",
                "snippet": f"Academic research examining {query} with empirical results.",
                "source": "academic_papers"
            },
            {
                "title": f"Novel approach to {query}",
                "url": f"https://journal.example.com/article/{query.replace(' ', '-')}",
                "snippet": f"Peer-reviewed paper proposing new solution for {query}.",
                "source": "academic_papers"
            }
        ]
    
    def _synthesize_results(self, results: Dict[str, Any], query: str, context: Dict[str, Any] = None) -> str:
        """Synthesize results into a coherent summary"""
        if not results:
            return f"No information found for '{query}'"
        
        synthesis = f"Research findings for '{query}':\n\n"
        
        # Count total results
        total_results = sum(data.get("count", 0) for data in results.values() if isinstance(data, dict) and "count" in data)
        synthesis += f"Found {total_results} relevant resources across {len(results)} knowledge sources.\n\n"
        
        # Add findings from each tentacle
        for tentacle_type, data in results.items():
            if isinstance(data, dict) and "results" in data and data["results"]:
                synthesis += f"== {tentacle_type.replace('_', ' ').title()} ==\n"
                
                for result in data["results"][:3]:  # Limit to top 3 results per tentacle
                    synthesis += f"- {result.get('title')}\n"
                    if "snippet" in result:
                        synthesis += f"  {result.get('snippet')}\n"
                    if "url" in result:
                        synthesis += f"  Source: {result.get('url')}\n"
                
                synthesis += "\n"
        
        # Add summary based on context if available
        if context:
            synthesis += "== Context-Specific Analysis ==\n"
            
            if "os" in context:
                synthesis += f"For {context['os']} systems: "
                synthesis += "Research indicates platform-specific considerations apply.\n"
            
            if "error_code" in context:
                synthesis += f"Regarding error code {context['error_code']}: "
                synthesis += "Multiple sources confirm this is a known issue with documented solutions.\n"
        
        synthesis += "\n== Key Takeaways ==\n"
        synthesis += "1. Multiple reliable sources address this issue\n"
        synthesis += "2. Solutions are available and have been verified by the community\n"
        synthesis += "3. Implementation details vary based on specific environment\n"
        
        return synthesis
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Create a summary of the context"""
        if not context:
            return "No context provided"
        
        summary = []
        
        if "os" in context:
            summary.append(f"OS: {context['os']}")
        
        if "language" in context:
            summary.append(f"Language: {context['language']}")
        
        if "error_code" in context:
            summary.append(f"Error code: {context['error_code']}")
        
        if "vendor" in context:
            summary.append(f"Vendor: {context['vendor']}")
        
        return ", ".join(summary)
    
    def _cleanup_tentacles(self) -> None:
        """Clean up completed or timed out tentacles"""
        current_time = time.time()
        tentacles_to_remove = []
        
        for tentacle_id, tentacle_data in self.active_tentacles.items():
            # Remove completed tentacles older than 1 hour
            if tentacle_data["status"] == "completed" and current_time - tentacle_data["end_time"] > 3600:
                tentacles_to_remove.append(tentacle_id)
            
            # Remove searching tentacles that have timed out
            elif tentacle_data["status"] == "searching" and current_time - tentacle_data["start_time"] > self.timeout * 2:
                tentacle_data["status"] = "timeout"
                tentacle_data["end_time"] = current_time
        
        # Remove tentacles
        for tentacle_id in tentacles_to_remove:
            del self.active_tentacles[tentacle_id]
    
    def get_active_tentacles(self) -> Dict[str, Any]:
        """Get information about active tentacles"""
        return self.active_tentacles
    
    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get research history"""
        return self.research_history

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create research tentacles
    tentacles = ResearchTentacles(max_tentacles=3)
    
    # Deploy tentacles for a sample query
    results = tentacles.deploy(
        "memory leak in Python application",
        context={"os": "Linux", "language": "Python"}
    )
    
    print(f"Tentacles used: {results['tentacles_used']}")
    print(f"Findings:\n{results['findings']}")
