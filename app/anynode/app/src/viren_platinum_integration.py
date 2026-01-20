#!/usr/bin/env python3
"""
Viren Platinum Integration - Integrates Document Suite with Platinum Edition
"""

import os
import logging
import gradio as gr
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("VirenPlatinumIntegration")

def integrate_document_suite(platinum_interface: gr.Blocks) -> gr.Blocks:
    """
    Integrate the document suite into the Platinum interface
    
    Args:
        platinum_interface: The Platinum interface
        
    Returns:
        Updated interface with document suite integrated
    """
    try:
        # Import document suite
        from viren_document_suite import VirenDocumentSuite
        doc_suite = VirenDocumentSuite()
        
        # Get document suite interface components
        doc_interface = doc_suite.create_interface()
        
        # Extract document suite tabs
        doc_tabs = None
        for component in doc_interface.blocks.values():
            if isinstance(component, gr.Tabs):
                doc_tabs = component
                break
        
        if not doc_tabs:
            logger.error("Could not find document suite tabs")
            return platinum_interface
        
        # Find the main tabs in the platinum interface
        platinum_tabs = None
        for component in platinum_interface.blocks.values():
            if isinstance(component, gr.Tabs):
                platinum_tabs = component
                break
        
        if not platinum_tabs:
            logger.error("Could not find platinum tabs")
            return platinum_interface
        
        # Add document suite tabs to platinum tabs
        for tab in doc_tabs:
            platinum_tabs.add_tab(tab)
        
        logger.info("Document suite integrated successfully")
        return platinum_interface
    except Exception as e:
        logger.error(f"Error integrating document suite: {e}")
        return platinum_interface

def create_integrated_interface() -> gr.Blocks:
    """
    Create an integrated interface with Platinum and Document Suite
    
    Returns:
        Integrated interface
    """
    try:
        # Import platinum interface
        from viren_platinum_interface import create_main_interface
        platinum_interface = create_main_interface()
        
        # Integrate document suite
        integrated_interface = integrate_document_suite(platinum_interface)
        
        return integrated_interface
    except Exception as e:
        logger.error(f"Error creating integrated interface: {e}")
        
        # Fallback to basic interface
        with gr.Blocks(title="Viren Platinum with Document Suite") as fallback_interface:
            gr.Markdown("# Viren Platinum with Document Suite")
            gr.Markdown("Error creating integrated interface. Please check logs.")
        
        return fallback_interface

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create integrated interface
    interface = create_integrated_interface()
    
    # Launch interface
    interface.launch()
