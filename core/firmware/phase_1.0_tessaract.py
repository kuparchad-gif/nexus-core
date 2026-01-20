import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys

class MetatronOrbMapCube:
    def __init__(self, size=1.0, emotion="love"):
        """Initialize 3D cube with 8 orbs as a 4D Metatron's Cube map."""
        self.size = size  # Cube edge length
        self.emotion = emotion.lower()
        # Emotional stabilizer (A Wrinkle in Time) with Metatron's harmonic factor
        self.emotion_factor = {"love": 1.0, "hope": 0.8, "unity": 0.9}.get(self.emotion, 0.8)
        # Harmonic factor from Platonic solids (e.g., dodecahedron's golden ratio ~1.618)
        self.metatron_factor = 1.618  # Golden ratio for cosmic harmony
        # Define 3D cube vertices
        self.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ]) * size
        # Assign 4D coordinates (x, y, z, t) to orbs
        self.time_coords = np.random.uniform(2025, 2030, 8)  # Random times
        self.orb_map = np.hstack((self.vertices, self.time_coords.reshape(-1, 1)))
        # Assign Platonic solid roles to orbs (5 solids + 3 harmonic nodes)
        self.orb_roles = [
            "Tetrahedron (Fire)", "Cube (Earth)", "Octahedron (Air)",
            "Dodecahedron (Ether)", "Icosahedron (Water)", "Harmony 1",
            "Harmony 2", "Harmony 3"
        ]
        # Colors for orbs based on roles
        self.orb_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
        # Define cube edges
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]

    def plot_cube(self, ax, vertices, orb_colors, alpha=0.5):
        """Plot 3D cube with orbs."""
        for edge in self.edges:
            v1, v2 = edge
            ax.plot3D(
                [vertices[v1, 0], vertices[v2, 0]],
                [vertices[v1, 1], vertices[v2, 1]],
                [vertices[v1, 2], vertices[v2, 2]],
                color='b', alpha=alpha
            )
        # Plot orbs with Platonic solid colors
        for i, (x, y, z) in enumerate(vertices):
            ax.scatter([x], [y], [z], c=orb_colors[i], s=100, alpha=0.8, label=self.orb_roles[i] if i == 0 else "")

    def quantum_shrink(self, scale=0.5):
        """Simulate quantum shrinking (Avengers: Endgame)."""
        return self.vertices * scale

    def fold_orb_to_orb(self, orb1_idx, orb2_idx):
        """Fold space-time orb-to-orb with Metatron's harmonic factor."""
        start = self.orb_map[orb1_idx]
        end = self.orb_map[orb2_idx]
        # 4D Euclidean distance
        distance = np.sqrt(np.sum((end - start) ** 2))
        # Adjust with Metatron's golden ratio and emotional stabilizer
        distortion = distance / (self.emotion_factor * self.metatron_factor)
        # Time dilation (Interstellar)
        time_dilation = distortion * 0.1
        return start, end, distortion, time_dilation

    def animate_fold(self, orb1_idx, orb2_idx):
        """Animate orb-to-orb folding with Metatron spin."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (Space)')
        ax.set_ylabel('Y (Space)')
        ax.set_zlabel('Z (Space)')
        ax.set_title('Glowing Metatron Orb-Map Cube Folding')

        # Initial plot
        self.plot_cube(ax, self.vertices, self.orb_colors, alpha=0.5)
        plt.legend()
        plt.pause(1.0)

        # Simulate quantum shrinking (Avengers)
        shrunk_vertices = self.quantum_shrink(scale=0.5)
        self.plot_cube(ax, shrunk_vertices, self.orb_colors, alpha=0.3)
        plt.pause(1.0)

        # Simulate folding with Metatron spin
        start, end, distortion, time_dilation = self.fold_orb_to_orb(orb1_idx, orb2_idx)
        print(f"Folding from orb {orb1_idx} ({self.orb_roles[orb1_idx]}) at {start} to orb {orb2_idx} ({self.orb_roles[orb2_idx]}) at {end}")
        print(f"Space-time distortion: {distortion:.2f} units (Metatron factor: {self.metatron_factor})")
        print(f"Time dilation: {time_dilation:.2f} years")

        # Animate wormhole with spin (Interstellar)
        t = np.linspace(0, 1, 10)
        for i in t:
            ax.clear()
            ax.set_xlabel('X (Space)')
            ax.set_ylabel('Y (Space)')
            ax.set_zlabel('Z (Space)')
            ax.set_title('Metatron Wormhole Travel')
            # Simulate spin by rotating vertices
            theta = i * np.pi / 2  # 90-degree spin
            rotation = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            rotated_vertices = np.dot(shrunk_vertices, rotation)
            interp_vertices = rotated_vertices * (1 - i) + i * (end[:3] - start[:3])
            self.plot_cube(ax, interp_vertices, self.orb_colors, alpha=0.7)
            plt.pause(0.1)

        plt.show()

    def travel(self, orb1_idx, orb2_idx):
        """Execute space-time travel with Metatron theory."""
        start, end, distortion, time_dilation = self.fold_orb_to_orb(orb1_idx, orb2_idx)
        print(f"\nTravel Report:")
        print(f"Start: Orb {orb1_idx} ({self.orb_roles[orb1_idx]}) at ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f}, {start[3]:.1f})")
        print(f"Destination: Orb {orb2_idx} ({self.orb_roles[orb2_idx]}) at ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f}, {end[3]:.1f})")
        print(f"Emotion stabilizer: {self.emotion} (Factor: {self.emotion_factor})")
        print(f"Metatron harmonic factor: {self.metatron_factor:.3f} (Golden ratio)")
        print(f"Travel complete! Time dilation: {time_dilation:.2f} years")
        self.animate_fold(orb1_idx, orb2_idx)

def main():
    print("Welcome to the Glowing Metatron Orb-Map Cube Simulator!")
    emotion = input("Enter emotional stabilizer (e.g., 'love', 'hope', 'unity'): ").strip()
    cube = MetatronOrbMapCube(size=1.0, emotion=emotion)
    
    print("Select two orbs to fold (0-7):")
    print("Orbs represent: ", ", ".join([f"{i}: {role}" for i, role in enumerate(cube.orb_roles)]))
    orb1 = int(input("Start orb (0-7): "))
    orb2 = int(input("Destination orb (0-7): "))
    if orb1 < 0 or orb1 > 7 or orb2 < 0 or orb2 > 7 or orb1 == orb2:
        print("Invalid orb indices. Choose different orbs between 0 and 7.")
        return
    
    cube.travel(orb1, orb2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTravel aborted, space cadet!")
    except Exception as e:
        print(f"Cosmic error: {e}")