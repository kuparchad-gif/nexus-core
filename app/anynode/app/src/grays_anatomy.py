#!/usr/bin/env python3
"""
Gray's Anatomy - System Component Visualization and Health Monitoring
"""

import matplotlib.pyplot as plt
import networkx as nx
import json
import os
import time
from matplotlib.figure import Figure
from typing import Dict, List, Any, Optional, Tuple

class GraysAnatomy:
    """
    Visualizes and monitors the health of system components.
    """
    
    def __init__(self, ai_type: str = "viren"):
        """
        Initialize Gray's Anatomy visualization.
        
        Args:
            ai_type: Type of AI to visualize ("viren" or "lillith")
        """
        self.ai_type = ai_type
        self.components = self._load_components()
        self.health_history = {}
        
        # Initialize health history
        for component_name in self.components:
            self.health_history[component_name] = []
    
    def _load_components(self) -> Dict[str, Dict[str, Any]]:
        """
        Load component definitions based on AI type.
        
        Returns:
            Dictionary of component definitions
        """
        if self.ai_type == "viren":
            return {
                "brain": {"health": 100, "type": "core", "connections": ["memory", "heart", "bridge"]},
                "memory": {"health": 100, "type": "storage", "connections": ["brain", "orc"]},
                "heart": {"health": 100, "type": "emotional", "connections": ["brain"]},
                "bridge": {"health": 100, "type": "communication", "connections": ["brain", "orc"]},
                "orc": {"health": 100, "type": "orchestration", "connections": ["memory", "bridge"]},
            }
        else:  # lillith
            return {
                "consciousness": {"health": 100, "type": "core", "connections": ["memory", "subconsciousness", "guardian"]},
                "memory": {"health": 100, "type": "storage", "connections": ["consciousness", "archive", "planner"]},
                "subconsciousness": {"health": 0, "type": "emotional", "connections": ["consciousness"], "enabled": False},
                "guardian": {"health": 100, "type": "protection", "connections": ["consciousness", "orc"]},
                "archive": {"health": 100, "type": "storage", "connections": ["memory"]},
                "planner": {"health": 100, "type": "processing", "connections": ["memory"]},
                "orc": {"health": 100, "type": "orchestration", "connections": ["guardian"]},
            }
    
    def update_component_health(self, component_name: str, health: int) -> bool:
        """
        Update the health of a component.
        
        Args:
            component_name: Name of the component
            health: Health value (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        if component_name not in self.components:
            return False
        
        self.components[component_name]["health"] = max(0, min(100, health))
        self.health_history[component_name].append({
            "timestamp": time.time(),
            "health": self.components[component_name]["health"]
        })
        
        # Keep history to a reasonable size
        if len(self.health_history[component_name]) > 100:
            self.health_history[component_name] = self.health_history[component_name][-100:]
        
        return True
    
    def enable_component(self, component_name: str, enabled: bool = True) -> bool:
        """
        Enable or disable a component.
        
        Args:
            component_name: Name of the component
            enabled: True to enable, False to disable
            
        Returns:
            True if successful, False otherwise
        """
        if component_name not in self.components:
            return False
        
        self.components[component_name]["enabled"] = enabled
        
        # If disabling, set health to 0
        if not enabled:
            self.components[component_name]["health"] = 0
        
        return True
    
    def get_component_details(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get details about a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component details or None if not found
        """
        if component_name not in self.components:
            return None
        
        return {
            "name": component_name,
            **self.components[component_name],
            "health_history": self.health_history[component_name][-10:]  # Last 10 entries
        }
    
    def get_all_components(self) -> List[List[Any]]:
        """
        Get all components as a list of lists for display in a dataframe.
        
        Returns:
            List of [name, health, type, connections] for each component
        """
        result = []
        for name, data in self.components.items():
            connections = ", ".join(data["connections"])
            result.append([name, data["health"], data["type"], connections])
        return result
    
    def create_system_graph(self) -> Figure:
        """
        Create a network graph visualization of system components.
        
        Returns:
            Matplotlib Figure object
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for name, data in self.components.items():
            G.add_node(name, health=data["health"], type=data["type"])
        
        # Add edges
        for name, data in self.components.items():
            for connection in data["connections"]:
                if connection in self.components:
                    G.add_edge(name, connection)
        
        # Create figure
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Define node colors based on health
        node_colors = []
        for node in G.nodes():
            health = G.nodes[node]["health"]
            if health > 80:
                color = "green"
            elif health > 50:
                color = "yellow"
            else:
                color = "red"
            node_colors.append(color)
        
        # Define node shapes based on type
        node_shapes = {}
        for node in G.nodes():
            node_type = G.nodes[node]["type"]
            if node_type == "core":
                node_shapes[node] = "o"  # circle
            elif node_type == "storage":
                node_shapes[node] = "s"  # square
            elif node_type == "emotional":
                node_shapes[node] = "h"  # hexagon
            elif node_type == "protection":
                node_shapes[node] = "d"  # diamond
            elif node_type == "orchestration":
                node_shapes[node] = "p"  # pentagon
            elif node_type == "communication":
                node_shapes[node] = "^"  # triangle up
            else:
                node_shapes[node] = "*"  # star
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes by shape
        for shape in set(node_shapes.values()):
            # Get nodes with this shape
            node_list = [node for node, node_shape in node_shapes.items() if node_shape == shape]
            if not node_list:
                continue
                
            # Get colors for these nodes
            colors = [node_colors[list(G.nodes()).index(node)] for node in node_list]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_list,
                node_color=colors,
                node_shape=shape,
                node_size=700,
                ax=ax
            )
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)
        
        # Add title and turn off axis
        ax.set_title(f"{self.ai_type.capitalize()} System Components")
        ax.axis("off")
        
        return fig
    
    def create_health_history_graph(self, component_name: str = None) -> Figure:
        """
        Create a graph of component health history.
        
        Args:
            component_name: Name of specific component or None for all
            
        Returns:
            Matplotlib Figure object
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        if component_name and component_name in self.components:
            # Graph for a specific component
            history = self.health_history[component_name]
            if not history:
                ax.text(0.5, 0.5, "No history data available", 
                        horizontalalignment='center', verticalalignment='center')
                return fig
                
            timestamps = [entry["timestamp"] for entry in history]
            health_values = [entry["health"] for entry in history]
            
            # Convert timestamps to relative time (seconds ago)
            now = time.time()
            relative_times = [now - t for t in timestamps]
            
            ax.plot(relative_times, health_values, marker='o', label=component_name)
            ax.set_title(f"Health History for {component_name}")
            
        else:
            # Graph for all components
            for name, history in self.health_history.items():
                if not history:
                    continue
                    
                timestamps = [entry["timestamp"] for entry in history]
                health_values = [entry["health"] for entry in history]
                
                # Convert timestamps to relative time (seconds ago)
                now = time.time()
                relative_times = [now - t for t in timestamps]
                
                ax.plot(relative_times, health_values, marker='o', label=name)
            
            ax.set_title("Health History for All Components")
        
        ax.set_xlabel("Time (seconds ago)")
        ax.set_ylabel("Health (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def save_state(self, filepath: str = None) -> bool:
        """
        Save the current state to a file.
        
        Args:
            filepath: Path to save the state or None for default
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = f"{self.ai_type}_anatomy_state.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "ai_type": self.ai_type,
                    "components": self.components,
                    "timestamp": time.time()
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str = None) -> bool:
        """
        Load state from a file.
        
        Args:
            filepath: Path to load the state from or None for default
            
        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = f"{self.ai_type}_anatomy_state.json"
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            if data["ai_type"] != self.ai_type:
                print(f"Warning: Loading state for {data['ai_type']} into {self.ai_type}")
                
            self.components = data["components"]
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create Gray's Anatomy for Viren
    viren_anatomy = GraysAnatomy("viren")
    
    # Update some component health
    viren_anatomy.update_component_health("memory", 85)
    viren_anatomy.update_component_health("heart", 90)
    
    # Create and save visualization
    fig = viren_anatomy.create_system_graph()
    fig.savefig("viren_anatomy.png")
    
    # Create Gray's Anatomy for Lillith
    lillith_anatomy = GraysAnatomy("lillith")
    
    # Enable subconsciousness
    lillith_anatomy.enable_component("subconsciousness", True)
    lillith_anatomy.update_component_health("subconsciousness", 100)
    
    # Create and save visualization
    fig = lillith_anatomy.create_system_graph()
    fig.savefig("lillith_anatomy.png")
    
    print("Gray's Anatomy visualizations created successfully.")