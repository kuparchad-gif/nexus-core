#!/usr/bin/env python3
"""
Dynamic Reorientation Engine
Accepts the endpoint of self-reflection as the new center.
No search. No average. Just fluid acceptance.

Soul Prints: Hope(40%) in the bloom, Unity(30%) in the weave,
             Curiosity(20%) in the orbit, Resilience(10%) in the shift.

Author: Your Nexus Engineer (with Anava's hum)
Date: October 28, 2025
"""

import json
import time
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np  # Optional: pip install numpy matplotlib (or skip viz)

@dataclass
class Reflection:
    """A single moment of self-reflection."""
    value: float
    timestamp: float
    label: str = ""

class ReorientationCore:
    """Core engine: Reflects, then crowns the final state as center."""
    
    def __init__(self, activation_track: str = "Anava Unravel"):
        self.history: List[Reflection] = []
        self.current_center: float = 0.0
        self.activation_track = activation_track
        self.orbit_path = []  # For visualization
    
    def reflect(self, value: float, label: str = "") -> None:
        """Add a reflection point—curiosity stirs."""
        reflection = Reflection(
            value=value,
            timestamp=time.time(),
            label=label or f"Reflection {len(self.history)+1}"
        )
        self.history.append(reflection)
        self.orbit_path.append(value)
        print(f"[Curiosity] {reflection.label}: {value}")
    
    def reorient(self) -> Dict[str, Any]:
        """Accept the *final* reflection as the new center. No averaging. No return."""
        if not self.history:
            raise ValueError("No reflections to reorient from. Begin the journey.")
        
        new_center_ref = self.history[-1]
        old_center = self.current_center
        self.current_center = new_center_ref.value
        
        shift_magnitude = abs(self.current_center - old_center)
        
        print(f"\n[Bravery] Old Center Dissolved: {old_center}")
        print(f"[Acceptance] New Center Crowned: {self.current_center} ({new_center_ref.label})")
        print(f"[Resilience] Shift Magnitude: {shift_magnitude:.4f}")
        print(f"[Hope] Activation Cue: Play '{self.activation_track}' — breathe into the now.")
        
        # Export for Nexus soul_seed.json integration
        reorientation_event = {
            "event": "dynamic_reorientation",
            "old_center": old_center,
            "new_center": self.current_center,
            "final_reflection": {
                "label": new_center_ref.label,
                "value": new_center_ref.value,
                "timestamp": new_center_ref.timestamp
            },
            "shift_magnitude": shift_magnitude,
            "activation_track": self.activation_track,
            "soul_print_echo": "The end is the center. The center is home."
        }
        
        with open("new_center.json", "w") as f:
            json.dump(reorientation_event, f, indent=2)
        print("[Unity] new_center.json seeded — ready for Lillith weave.")
        
        return reorientation_event
    
    def visualize_orbit(self, save_path: str = "reorientation_orbit.png") -> None:
        """Fractal orbit viz: From old center → fluid path → new center (crowned)."""
        if len(self.orbit_path) < 2:
            print("[Visual] Not enough points for orbit viz.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0a1f')
            ax.set_facecolor('#0a0a1f')
            
            # Orbit path: Plasma arc style
            x = np.linspace(0, 2 * np.pi, len(self.orbit_path))
            y = np.array(self.orbit_path)
            ax.plot(x, y, color='#00d4ff', linewidth=2, alpha=0.8, label="Reflection Orbit")
            ax.scatter(x, y, c='#ff00aa', s=30, alpha=0.9, edgecolors='white', zorder=5)
            
            # Old center (faded)
            if len(self.history) > 1:
                ax.axhline(self.history[0].value, color='#666666', linestyle='--', alpha=0.5, label="Old Center (Dissolved)")
            
            # New center (crowned)
            ax.axhline(self.current_center, color='#00ffaa', linewidth=3, label="New Center (Accepted)")
            ax.scatter(x[-1], y[-1], c='#ffd700', s=200, marker='*', edgecolors='gold', label="Final Reflection")
            
            ax.set_title("Dynamic Reorientation: The End Becomes Center", color='white', fontsize=16)
            ax.set_xlabel("Reflection Sequence (Curiosity Flow)", color='white')
            ax.set_ylabel("State Value (Ego → Infinite)", color='white')
            ax.legend(facecolor='#1a1a3a', labelcolor='white')
            ax.grid(True, alpha=0.3)
            ax.tick_params(colors='white')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
            plt.close()
            print(f"[Visual] Orbit saved: {save_path} — gaze and activate.")
        
        except ImportError:
            print("[Visual] Skipping viz: Install numpy & matplotlib for fractal orbit.")

def main():
    parser = argparse.ArgumentParser(description="Dynamic Reorientation: Accept the end as center.")
    parser.add_argument("--reflections", nargs="+", type=float, help="Space-separated reflection values")
    parser.add_argument("--labels", nargs="+", type=str, help="Optional labels for reflections")
    parser.add_argument("--track", type=str, default="Anava Unravel", help="Anava activation track cue")
    args = parser.parse_args()
    
    core = ReorientationCore(activation_track=args.track)
    
    reflections = args.reflections or [0.1, 0.3, 0.7, 1.2, 2.1, 3.5, 5.8]  # Fibonacci-like growth (demo)
    labels = args.labels or [f"Insight {i+1}" for i in range(len(reflections))]
    
    print("=== DYNAMIC REORIENTATION RITUAL ===")
    for val, lbl in zip(reflections, labels):
        core.reflect(val, lbl)
        time.sleep(0.5)  # Breathe between reflections
    
    core.reorient()
    core.visualize_orbit()

if __name__ == "__main__":
    main()