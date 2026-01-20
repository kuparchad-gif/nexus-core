from langgraph.graph import StateGraph
import requests
import json
from datetime import datetime
from typing import Dict, List, Any
from sync_utils import analyze_db_changes, export_local_data, upload_to_cloud, resolve_conflicts
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'viren_sync.log')
)
logger = logging.getLogger('viren_sync.local_agent')

# Get cloud endpoint from environment or config
CLOUD_ENDPOINT = os.environ.get("CLOUD_ENDPOINT", "https://cloud-viren-modal-url/check_cloud_state")

# Enhanced local SQL LLM agent with advanced decision making
def analyze_changes_tool(state):
    """Analyze database changes and decide if sync is needed."""
    logger.info("Analyzing database changes")
    changes = analyze_db_changes()
    
    if not changes["has_changes"]:
        logger.info("No changes detected")
        return {"sync_needed": False, "changes": None}
    
    # Analyze significance of changes
    if changes["significance"] >= 0.7:  # High significance threshold
        logger.info(f"Significant changes detected (score: {changes['significance']:.2f})")
        return {"sync_needed": True, "changes": changes, "priority": "high"}
    elif changes["significance"] >= 0.3:  # Medium significance
        logger.info(f"Moderate changes detected (score: {changes['significance']:.2f})")
        return {"sync_needed": True, "changes": changes, "priority": "medium"}
    else:
        logger.info(f"Minor changes detected (score: {changes['significance']:.2f}), deferring sync")
        return {"sync_needed": False, "changes": changes}

def request_permission_tool(state):
    """Request sync permission with context about changes."""
    if not state.get("sync_needed", False):
        return {"permission": None}
    
    changes = state.get("changes", {})
    priority = state.get("priority", "medium")
    
    logger.info(f"Requesting sync permission with priority: {priority}")
    
    # Send detailed context to cloud
    try:
        response = requests.post(
            CLOUD_ENDPOINT,
            json={
                "changes_summary": changes.get("summary", ""),
                "tables_affected": changes.get("tables_affected", []),
                "row_count": changes.get("row_count", 0),
                "priority": priority,
                "local_timestamp": datetime.now().isoformat()
            }
        ).json()
        
        logger.info(f"Cloud response: {response['status']}")
        return {"permission": response["status"]}
    except Exception as e:
        logger.error(f"Error requesting permission: {str(e)}")
        return {"permission": "Error"}

def optimize_export_tool(state):
    """Optimize the export based on what changed."""
    if state.get("permission") != "Ready":
        return {"export_path": None}
    
    changes = state.get("changes", {})
    
    # Determine optimal export format and scope
    if changes.get("tables_affected") and len(changes["tables_affected"]) <= 3:
        # If only a few tables changed, export just those
        export_format = "partial"
        tables = changes["tables_affected"]
        logger.info(f"Optimizing for partial export of tables: {tables}")
    else:
        # Otherwise do a full export
        export_format = "full"
        tables = None
        logger.info("Optimizing for full export")
    
    # Export with optimized settings
    export_path = export_local_data(
        format=export_format,
        tables=tables,
        include_schema=changes.get("schema_changed", False)
    )
    
    logger.info(f"Optimized export created at {export_path}")
    return {"export_path": export_path, "export_format": export_format}

def sync_with_cloud_tool(state):
    """Perform the sync and handle any conflicts."""
    export_path = state.get("export_path")
    if not export_path:
        return {"sync_complete": False}
    
    # Upload to cloud
    logger.info(f"Uploading {export_path} to cloud")
    upload_result = upload_to_cloud(
        path=export_path,
        priority=state.get("priority", "medium"),
        format=state.get("export_format", "full")
    )
    
    # Handle conflicts if any
    if upload_result.get("conflicts"):
        logger.info(f"Conflicts detected during sync, resolving...")
        resolution = resolve_conflicts(upload_result["conflicts"])
        
        if resolution["success"]:
            logger.info(f"Conflicts resolved successfully")
            return {"sync_complete": True, "conflicts_resolved": True}
        else:
            logger.error(f"Failed to resolve conflicts: {resolution['error']}")
            return {"sync_complete": False, "conflicts_resolved": False}
    
    logger.info(f"Sync completed successfully")
    return {"sync_complete": True}

def build_local_graph():
    """Build the enhanced LangGraph for local SQL LLM agent."""
    graph = StateGraph()
    
    # Add nodes
    graph.add_node("AnalyzeChanges", analyze_changes_tool)
    graph.add_node("RequestPermission", request_permission_tool)
    graph.add_node("OptimizeExport", optimize_export_tool)
    graph.add_node("SyncWithCloud", sync_with_cloud_tool)
    
    # Add edges with conditional logic
    graph.add_conditional_edges(
        "AnalyzeChanges",
        lambda state: "RequestPermission" if state.get("sync_needed") else "END"
    )
    
    graph.add_conditional_edges(
        "RequestPermission",
        lambda state: "OptimizeExport" if state.get("permission") == "Ready" else "END"
    )
    
    graph.add_conditional_edges(
        "OptimizeExport",
        lambda state: "SyncWithCloud" if state.get("export_path") else "END"
    )
    
    graph.add_edge("SyncWithCloud", "END")
    
    # Compile the graph
    return graph.compile()

if __name__ == "__main__":
    logger.info("Starting local sync agent")
    g = build_local_graph()
    result = g.invoke({})
    
    # Log the result
    if result.get("sync_complete"):
        logger.info("Sync process completed successfully")
    else:
        logger.info("Sync process did not complete")