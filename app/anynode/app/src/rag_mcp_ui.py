#!/usr/bin/env python3
"""
RAG MCP UI for Cloud Viren
Provides a web interface for the RAG Master Control Program
"""

import os
import sys
import json
import time
import logging
import threading
import webbrowser
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGMCPUi")

try:
    import gradio as gr
except ImportError:
    logger.warning("Gradio not installed. Install with: pip install gradio")
    logger.warning("Continuing in limited mode...")
    gr = None

# Import RAG MCP Controller
try:
    from rag_mcp_controller import RAGMCPController
except ImportError:
    logger.error("RAG MCP Controller module not found. Make sure rag_mcp_controller.py is in the same directory.")
    sys.exit(1)

# Import theme colors
THEME_COLORS = {
    "plumb": "#A2799A",    # Rich purple
    "primer": "#93AEC5",   # Medium blue
    "silver": "#AFC5DC",   # Light blue
    "putty": "#C6D6E2",    # Very light blue
    "dried_putty": "#D8E3EB",  # Pale blue
    "white": "#EBF2F6"     # Off-white
}

class RAGMCPUi:
    """
    RAG MCP UI for Cloud Viren
    Provides a web interface for the RAG Master Control Program
    """
    
    def __init__(self, controller: RAGMCPController = None):
        """Initialize the RAG MCP UI"""
        self.controller = controller or RAGMCPController()
        self.interface = None
        
        logger.info("RAG MCP UI initialized")
    
    def start(self, port: int = 7860, share: bool = False, inbrowser: bool = True) -> None:
        """Start the RAG MCP UI"""
        if not gr:
            logger.error("Gradio not installed. Cannot start UI.")
            return
        
        logger.info(f"Starting RAG MCP UI on port {port}")
        
        # Start controller if not already running
        if not self.controller.running:
            self.controller.start()
        
        # Create interface
        with gr.Blocks(theme=self._create_theme()) as interface:
            gr.Markdown("# Cloud Viren RAG MCP")
            gr.Markdown("Retrieval-Augmented Generation Master Control Program")
            
            with gr.Tab("Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Query",
                            placeholder="Enter your query here...",
                            lines=3
                        )
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="Model Size",
                                choices=["1B", "3B", "7B", "14B", "27B", "128B", "256B"],
                                value="3B"
                            )
                            
                            submit_button = gr.Button("Submit", variant="primary")
                    
                    with gr.Column(scale=2):
                        with gr.Accordion("Advanced Options", open=False):
                            max_results = gr.Slider(
                                label="Max Results",
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1
                            )
                            
                            min_score = gr.Slider(
                                label="Min Score",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.05
                            )
                            
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.05
                            )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        response_output = gr.Textbox(
                            label="Response",
                            lines=10,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        with gr.Accordion("Retrieved Documents", open=True):
                            docs_output = gr.JSON(label="Documents")
                
                with gr.Row():
                    timing_output = gr.JSON(label="Timing")
            
            with gr.Tab("Knowledge"):
                with gr.Row():
                    with gr.Column(scale=3):
                        knowledge_input = gr.Textbox(
                            label="Knowledge Text",
                            placeholder="Enter knowledge to add...",
                            lines=5
                        )
                        
                        with gr.Row():
                            collection_dropdown = gr.Dropdown(
                                label="Collection",
                                choices=["knowledge", "models", "diagnostics"],
                                value="knowledge"
                            )
                            
                            add_button = gr.Button("Add Knowledge", variant="primary")
                    
                    with gr.Column(scale=2):
                        with gr.Accordion("Metadata", open=True):
                            source_input = gr.Textbox(
                                label="Source",
                                placeholder="Where this knowledge comes from"
                            )
                            
                            topic_input = gr.Textbox(
                                label="Topic",
                                placeholder="Topic or category"
                            )
                
                knowledge_result = gr.JSON(label="Result")
            
            with gr.Tab("Status"):
                refresh_button = gr.Button("Refresh Status")
                status_output = gr.JSON(label="Status")
            
            # Set up event handlers
            submit_button.click(
                fn=self._process_query,
                inputs=[query_input, model_dropdown, max_results, min_score, temperature],
                outputs=[response_output, docs_output, timing_output]
            )
            
            add_button.click(
                fn=self._add_knowledge,
                inputs=[knowledge_input, collection_dropdown, source_input, topic_input],
                outputs=[knowledge_result]
            )
            
            refresh_button.click(
                fn=self._get_status,
                inputs=[],
                outputs=[status_output]
            )
        
        # Start interface
        self.interface = interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            inbrowser=inbrowser
        )
    
    def _create_theme(self) -> gr.Theme:
        """Create a custom theme using company colors"""
        return gr.Theme(
            primary_hue=THEME_COLORS["plumb"],
            secondary_hue=THEME_COLORS["primer"],
            neutral_hue=THEME_COLORS["silver"],
            spacing_size=gr.themes.sizes.spacing_md,
            radius_size=gr.themes.sizes.radius_md,
            text_size=gr.themes.sizes.text_md
        )
    
    def _process_query(self, query: str, model_size: str, max_results: int, 
                      min_score: float, temperature: float) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Process a query using the RAG MCP Controller"""
        if not query:
            return "Please enter a query", [], {}
        
        # Set up options
        retrieval_options = {
            "max_results": max_results,
            "min_score": min_score
        }
        
        generation_options = {
            "temperature": temperature
        }
        
        # Process query
        result = self.controller.process_query(
            query=query,
            model_size=model_size,
            retrieval_options=retrieval_options,
            generation_options=generation_options
        )
        
        # Extract results
        response = result.get("response", "Error processing query")
        docs = result.get("retrieved_docs", [])
        timing = result.get("timing", {})
        
        return response, docs, timing
    
    def _add_knowledge(self, text: str, collection: str, source: str, topic: str) -> Dict[str, Any]:
        """Add knowledge to the vector database"""
        if not text:
            return {"status": "error", "message": "Please enter knowledge text"}
        
        # Set up metadata
        metadata = {}
        if source:
            metadata["source"] = source
        if topic:
            metadata["topic"] = topic
        
        # Add knowledge
        result = self.controller.add_knowledge(
            text=text,
            metadata=metadata,
            collection=collection
        )
        
        return result
    
    def _get_status(self) -> Dict[str, Any]:
        """Get RAG MCP status"""
        return self.controller.get_stats()

# Run the UI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG MCP UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the UI on")
    parser.add_argument("--share", action="store_true", help="Share the UI publicly")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the browser automatically")
    
    args = parser.parse_args()
    
    # Create controller
    controller = RAGMCPController()
    
    # Create UI
    ui = RAGMCPUi(controller)
    
    # Start UI
    ui.start(port=args.port, share=args.share, inbrowser=not args.no_browser)