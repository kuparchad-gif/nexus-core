#!/usr/bin/env python3
"""
Launcher for the Viren AI System (Complete Platinum Edition)
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

def launch_platinum_complete():
    """Launch the full integrated interface"""
    try:
        from viren_platinum_integration_complete import create_integrated_interface
        logging.info("Launching: Viren Platinum Complete")
        ui = create_integrated_interface()
        ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
    except ImportError as e:
        logging.error(f"Import failed: {e}")
        sys.exit(1)

def launch_platinum_core():
    """Launch the core Platinum interface (no doc suite or GitHub)"""
    try:
        from viren_platinum_interface import create_main_interface
        logging.info("Launching: Viren Platinum Core")
        ui = create_main_interface()
        ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
    except ImportError as e:
        logging.error(f"Import failed: {e}")
        sys.exit(1)

def launch_mcp():
    """Launch Mission Control Panel (MCP)"""
    try:
        from viren_platinum_mcp import create_interface
        logging.info("Launching: Viren Mission Control Panel (MCP)")
        ui = create_interface()
        ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
    except ImportError as e:
        logging.error(f"Import failed: {e}")
        sys.exit(1)

def launch_doc_suite():
    """Launch Document Suite only"""
    try:
        from viren_document_suite import VirenDocumentSuite
        logging.info("Launching: Document Suite")
        ui = VirenDocumentSuite().create_interface()
        ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
    except ImportError as e:
        logging.error(f"Document suite module not found: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Launch the Viren AI Engine")
    parser.add_argument(
        "--mode",
        choices=["complete", "platinum", "mcp", "docs"],
        default="complete",
        help="""
        Choose interface mode:
        - complete: Full integrated suite (Chat, Docs, GitHub, Router, MLX)
        - platinum: Chat + document studio + model lab (no GitHub/router)
        - mcp: Mission Control Panel (advanced ops mode)
        - docs: Document suite only
        """
    )

    args = parser.parse_args()

    if args.mode == "complete":
        launch_platinum_complete()
    elif args.mode == "platinum":
        launch_platinum_core()
    elif args.mode == "mcp":
        launch_mcp()
    elif args.mode == "docs":
        launch_doc_suite()
    else:
        logging.error("Invalid mode selected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
