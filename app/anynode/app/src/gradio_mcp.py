# Services/gradio_mcp.py
# Purpose: Gradio Mission Control Panel for Viren

import os
import sys
import logging
import threading
import json
import time
import asyncio
from typing import Dict, Any, List, Optional

# Import cloud models
from Config.modal.cloud_models import generate_text

# Configure logging
logger = logging.getLogger("gradio_mcp")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/gradio_mcp.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class GradioMCP:
    """
    Gradio Mission Control Panel for Viren.
    Provides a web-based GUI for monitoring and controlling Viren.
    """
    
    def __init__(self):
        """Initialize the Gradio MCP."""
        self.port = int(os.environ.get("GRADIO_PORT", 7860))
        self.app = None
        self.thread = None
        self.running = False
        
        # Check if Gradio is available
        try:
            import gradio as gr
            self.gradio_available = True
        except ImportError:
            logger.warning("Gradio not available, GUI will not be started")
            self.gradio_available = False
    
    def start(self):
        """Start the Gradio MCP in a separate thread."""
        if not self.gradio_available:
            logger.warning("Cannot start Gradio MCP: Gradio not available")
            return False
        
        if self.running:
            logger.warning("Gradio MCP already running")
            return True
        
        self.thread = threading.Thread(target=self._run_app)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started Gradio MCP on port {self.port}")
        return True
    
    def _run_app(self):
        """Run the Gradio app."""
        import gradio as gr
        
        # Get service status
        def get_service_status():
            from Services.self_management import self_management_service
            
            try:
                anatomy = self_management_service.get_system_anatomy()
                services = anatomy.get("services", {})
                
                status = []
                for name, info in services.items():
                    status.append({
                        "name": name,
                        "type": info.get("type", "unknown"),
                        "status": "active"  # We could add real status checking later
                    })
                
                return status
            except Exception as e:
                logger.error(f"Error getting service status: {e}")
                return []
        
        # Get model status
        def get_model_status():
            try:
                from config.model_config import load_model_config
                
                config = load_model_config()
                role_models = config.get("role_models", {})
                
                status = []
                for role, model in role_models.items():
                    if isinstance(model, dict):
                        for subrole, submodel in model.items():
                            status.append({
                                "role": f"{role}.{subrole}",
                                "model": submodel,
                                "status": "loaded"  # We could add real status checking later
                            })
                    else:
                        status.append({
                            "role": role,
                            "model": model,
                            "status": "loaded"  # We could add real status checking later
                        })
                
                # Add cloud models
                status.append({"role": "cloud", "model": "Cloud 1B", "status": "available"})
                status.append({"role": "cloud", "model": "Cloud 3B", "status": "available"})
                status.append({"role": "cloud", "model": "Cloud 7B", "status": "available"})
                
                return status
            except Exception as e:
                logger.error(f"Error getting model status: {e}")
                return []
        
        # Get approval requests
        def get_approval_requests():
            try:
                from Services.approval_system import approval_system
                
                requests = approval_system.list_approval_requests()
                return requests
            except Exception as e:
                logger.error(f"Error getting approval requests: {e}")
                return []
        
        # Approve a request
        def approve_request(request_id, guardian, comment):
            try:
                from Services.approval_system import approval_system
                
                success = approval_system.approve_request_by_guardian(
                    request_id=request_id,
                    guardian=guardian,
                    comment=comment
                )
                
                return f"{'Approved' if success else 'Failed to approve'} request {request_id}"
            except Exception as e:
                logger.error(f"Error approving request: {e}")
                return f"Error: {str(e)}"
        
        # Reject a request
        def reject_request(request_id, guardian, comment):
            try:
                from Services.approval_system import approval_system
                
                success = approval_system.reject_request_by_guardian(
                    request_id=request_id,
                    guardian=guardian,
                    comment=comment
                )
                
                return f"{'Rejected' if success else 'Failed to reject'} request {request_id}"
            except Exception as e:
                logger.error(f"Error rejecting request: {e}")
                return f"Error: {str(e)}"
        
        # Chat with Viren
        def chat_with_viren(message, history, model_choice="Local"):
            try:
                if not message:
                    return history
                
                # Use cloud models if selected
                if model_choice.startswith("Cloud"):
                    # Extract size from selection (1B, 3B, 7B)
                    size = model_choice.split(" ")[1]
                    response = asyncio.run(generate_text(message, model_size=size))
                else:
                    # Use the consciousness service for local models
                    from Services.consciousness_service import consciousness_service
                    result = consciousness_service.process_message(message)
                    response = result.get("response", "I'm unable to respond at the moment.")
                
                return history + [[message, response]]
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                return history + [[message, f"Error: {str(e)}"]]
        
        # New function for Gray's Anatomy panel
        def get_system_anatomy():
            from Services.self_management import self_management_service
            
            try:
                # Get the full system anatomy
                anatomy = self_management_service.get_system_anatomy()
                
                # Process core components
                core_components = []
                for name, info in anatomy.get("core_components", {}).items():
                    status = "Active" if os.path.exists(info.get("path", "")) else "Inactive"
                    core_components.append({
                        "component": name,
                        "description": info.get("description", "No description"),
                        "status": status,
                        "path": info.get("path", "Unknown")
                    })
                
                # Process services
                services = []
                for name, info in anatomy.get("services", {}).items():
                    status = "Active" if os.path.exists(info.get("path", "")) else "Inactive"
                    services.append({
                        "component": name,
                        "description": f"Service ({info.get('type', 'unknown')})",
                        "status": status,
                        "path": info.get("path", "Unknown")
                    })
                
                # Process bridges
                bridges = []
                for name, info in anatomy.get("bridges", {}).items():
                    status = "Active" if os.path.exists(info.get("path", "")) else "Inactive"
                    bridges.append({
                        "component": name,
                        "description": f"Bridge ({len(info.get('functions', []))} functions)",
                        "status": status,
                        "path": info.get("path", "Unknown")
                    })
                
                # Combine all components
                all_components = core_components + services + bridges
                
                # Sort by component name
                all_components.sort(key=lambda x: x["component"])
                
                # Convert to list format for DataFrame
                result = []
                for comp in all_components:
                    result.append([
                        comp["component"],
                        comp["description"],
                        comp["status"],
                        comp["path"]
                    ])
                
                return result
            except Exception as e:
                logger.error(f"Error getting system anatomy: {e}")
                return [["Error", f"Failed to load anatomy: {str(e)}", "Error", "N/A"]]
        
        # Function to get component details
        def get_component_details(component_name):
            try:
                from Services.self_management import self_management_service
                anatomy = self_management_service.get_system_anatomy()
                
                # Look in all sections
                for section in ["core_components", "services", "bridges"]:
                    if component_name in anatomy.get(section, {}):
                        return anatomy[section][component_name]
                
                return {"error": f"Component {component_name} not found"}
            except Exception as e:
                return {"error": str(e)}
        
        # Create the Gradio interface
        with gr.Blocks(title="Viren Mission Control Panel") as app:
            gr.Markdown("# Viren Mission Control Panel")
            
            with gr.Tab("Dashboard"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## System Status")
                        system_status = gr.DataFrame(
                            headers=["Component", "Status", "Details"],
                            value=[
                                ["Viren Core", "Active", "Running normally"],
                                ["Memory System", "Active", "Connected"],
                                ["Model Router", "Active", "Routing requests"]
                            ]
                        )
                        refresh_status = gr.Button("Refresh Status")
                    
                    with gr.Column():
                        gr.Markdown("## Active Models")
                        model_status = gr.DataFrame(
                            headers=["Role", "Model", "Status"],
                            value=get_model_status()
                        )
                        refresh_models = gr.Button("Refresh Models")
            
            # Add Gray's Anatomy tab
            with gr.Tab("Gray's Anatomy"):
                gr.Markdown("## Viren's Gray's Anatomy")
                gr.Markdown("This panel shows all components of Viren's system, their descriptions, and current status.")
                
                anatomy_table = gr.DataFrame(
                    headers=["Component", "Description", "Status", "Path"],
                    value=get_system_anatomy(),
                    height=600
                )
                
                refresh_anatomy = gr.Button("Refresh Anatomy")
                
                # Component details section
                gr.Markdown("## Component Details")
                selected_component = gr.Textbox(label="Selected Component")
                component_details = gr.JSON(label="Details", value={})
            
            with gr.Tab("Services"):
                gr.Markdown("## Service Management")
                service_status = gr.DataFrame(
                    headers=["Service", "Type", "Status"],
                    value=get_service_status()
                )
                refresh_services = gr.Button("Refresh Services")
            
            with gr.Tab("Approval Requests"):
                gr.Markdown("## Pending Approval Requests")
                approval_requests = gr.DataFrame(
                    headers=["ID", "Type", "Requester", "Status", "Created"],
                    value=[]
                )
                refresh_requests = gr.Button("Refresh Requests")
                
                with gr.Row():
                    request_id = gr.Textbox(label="Request ID")
                    guardian = gr.Textbox(label="Guardian Name")
                    comment = gr.Textbox(label="Comment")
                
                with gr.Row():
                    approve_btn = gr.Button("Approve")
                    reject_btn = gr.Button("Reject")
                
                result_text = gr.Textbox(label="Result")
            
            with gr.Tab("Chat"):
                # Add model selection dropdown
                model_choice = gr.Dropdown(
                    choices=["Local", "Cloud 1B", "Cloud 3B", "Cloud 7B"],
                    value="Local",
                    label="Model"
                )
                chatbot = gr.Chatbot()
                msg = gr.Textbox(label="Message")
                send = gr.Button("Send")
            
            # Set up event handlers
            refresh_status.click(lambda: [
                ["Viren Core", "Active", "Running normally"],
                ["Memory System", "Active", "Connected"],
                ["Model Router", "Active", "Routing requests"]
            ], outputs=system_status)
            
            refresh_models.click(get_model_status, outputs=model_status)
            refresh_services.click(get_service_status, outputs=service_status)
            refresh_requests.click(lambda: [
                [r["id"], r["operation_type"], r["requester"], r["status"], r["created_at"]]
                for r in get_approval_requests()
            ], outputs=approval_requests)
            
            # Gray's Anatomy event handlers
            refresh_anatomy.click(get_system_anatomy, outputs=anatomy_table)
            selected_component.change(get_component_details, inputs=selected_component, outputs=component_details)
            
            approve_btn.click(
                approve_request,
                inputs=[request_id, guardian, comment],
                outputs=result_text
            )
            
            reject_btn.click(
                reject_request,
                inputs=[request_id, guardian, comment],
                outputs=result_text
            )
            
            # Update the chat handler to include model selection
            send.click(chat_with_viren, inputs=[msg, chatbot, model_choice], outputs=[chatbot])
        
        # Launch the app
        self.app = app
        self.running = True
        
        try:
            app.launch(server_name="0.0.0.0", server_port=self.port, share=False)
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}")
            self.running = False
    
    def stop(self):
        """Stop the Gradio MCP."""
        if not self.running:
            logger.warning("Gradio MCP not running")
            return
        
        self.running = False
        if self.app:
            try:
                self.app.close()
            except Exception as e:
                logger.error(f"Error closing Gradio app: {e}")
        
        logger.info("Stopped Gradio MCP")

# Create a singleton instance
gradio_mcp = GradioMCP()
