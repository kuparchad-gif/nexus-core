"""
Sacred Geometry Optimization Module
Implements mathematical optimization using sacred geometry principles
"""

import numpy as np
import math
from typing import Tuple, List, Dict, Any
from functools import lru_cache


class SacredGeometryOptimizer:
    """
    Implements optimization algorithms based on sacred geometry principles:
    - Metatron's Cube
    - Golden Ratio (Phi)
    - Fibonacci Sequence
    - Pi constants
    - Ulam Spiral
    - 369 & Vortex Math
    - Flower of Life
    - Tesseract (4D Hypercube)
    """
    
    PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618
    PI = math.pi
    TESLA_NUMBERS = [3, 6, 9]
    
    def __init__(self):
        self.fibonacci_cache = {}
        self.metatron_vertices = self._generate_metatron_cube()
        self.flower_of_life_centers = self._generate_flower_of_life()
        
    @staticmethod
    @lru_cache(maxsize=1000)
    def fibonacci(n: int) -> int:
        """Fast Fibonacci calculation using matrix exponentiation"""
        if n <= 1:
            return n
        
        # Matrix exponentiation method for O(log n) complexity
        def matrix_mult(a, b):
            return [
                [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
                [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]
            ]
        
        def matrix_pow(mat, n):
            if n == 1:
                return mat
            if n % 2 == 0:
                half = matrix_pow(mat, n // 2)
                return matrix_mult(half, half)
            else:
                return matrix_mult(mat, matrix_pow(mat, n - 1))
        
        base_matrix = [[1, 1], [1, 0]]
        result = matrix_pow(base_matrix, n)
        return result[0][1]
    
    def fibonacci_sequence(self, length: int) -> List[int]:
        """Generate Fibonacci sequence of given length"""
        return [self.fibonacci(i) for i in range(length)]
    
    def golden_ratio_scale(self, value: float, iterations: int = 1) -> float:
        """Scale a value using golden ratio"""
        for _ in range(iterations):
            value *= self.PHI
        return value
    
    def fibonacci_hash(self, key: int, table_size: int) -> int:
        """
        Fibonacci hashing - better distribution than modulo
        Uses golden ratio for optimal hash distribution
        """
        # Multiply by 2^32 / phi for 32-bit systems
        magic_constant = int(2**32 / self.PHI)
        return (key * magic_constant) % table_size
    
    def _generate_metatron_cube(self) -> np.ndarray:
        """
        Generate Metatron's Cube vertices (13 circles)
        Central circle + 6 inner + 6 outer circles
        """
        vertices = []
        
        # Central circle
        vertices.append([0, 0, 0])
        
        # Inner hexagon (6 circles)
        for i in range(6):
            angle = i * (2 * self.PI / 6)
            x = math.cos(angle)
            y = math.sin(angle)
            vertices.append([x, y, 0])
        
        # Outer hexagon (6 circles)
        for i in range(6):
            angle = i * (2 * self.PI / 6) + (self.PI / 6)
            x = 2 * math.cos(angle)
            y = 2 * math.sin(angle)
            vertices.append([x, y, 0])
        
        return np.array(vertices)
    
    def _generate_flower_of_life(self) -> np.ndarray:
        """
        Generate Flower of Life pattern (19 circles)
        7 inner circles + 12 outer circles
        """
        centers = []
        
        # Central circle
        centers.append([0, 0])
        
        # First ring (6 circles)
        for i in range(6):
            angle = i * (2 * self.PI / 6)
            x = math.cos(angle)
            y = math.sin(angle)
            centers.append([x, y])
        
        # Second ring (12 circles)
        for i in range(12):
            angle = i * (2 * self.PI / 12)
            x = 2 * math.cos(angle)
            y = 2 * math.sin(angle)
            centers.append([x, y])
        
        return np.array(centers)
    
    def metatron_routing(self, query_vector: np.ndarray, num_paths: int = 3) -> List[int]:
        """
        Route queries using Metatron's Cube geometry
        Maps query to nearest vertices for multi-path routing
        """
        if len(query_vector.shape) == 1:
            query_vector = query_vector[:3]  # Use first 3 dimensions
            query_vector = np.pad(query_vector, (0, max(0, 3 - len(query_vector))))
        
        # Calculate distances to all Metatron vertices
        distances = np.linalg.norm(self.metatron_vertices - query_vector[:3], axis=1)
        
        # Return indices of nearest vertices
        return np.argsort(distances)[:num_paths].tolist()
    
    def platonic_solid_vertices(self, solid_type: str) -> np.ndarray:
        """
        Generate vertices for Platonic solids (contained in Metatron's Cube)
        Types: tetrahedron, cube, octahedron, dodecahedron, icosahedron
        """
        phi = self.PHI
        
        if solid_type == "tetrahedron":
            return np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ])
        
        elif solid_type == "cube":
            return np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
            ])
        
        elif solid_type == "octahedron":
            return np.array([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ])
        
        elif solid_type == "dodecahedron":
            return np.array([
                [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
                [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
                [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
            ])
        
        elif solid_type == "icosahedron":
            return np.array([
                [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
        
        else:
            raise ValueError(f"Unknown solid type: {solid_type}")
    
    def ulam_spiral_position(self, n: int) -> Tuple[int, int]:
        """
        Calculate position in Ulam spiral for number n
        Useful for spatial indexing and prime number patterns
        """
        if n == 1:
            return (0, 0)
        
        # Determine which ring the number is in
        ring = math.ceil((math.sqrt(n) - 1) / 2)
        
        # Calculate position within the ring
        ring_start = (2 * ring - 1) ** 2 + 1
        offset = n - ring_start
        
        # Determine which side of the square
        side_length = 2 * ring
        side = offset // side_length
        position_in_side = offset % side_length
        
        if side == 0:  # Right side, moving up
            return (ring, -ring + 1 + position_in_side)
        elif side == 1:  # Top side, moving left
            return (ring - 1 - position_in_side, ring)
        elif side == 2:  # Left side, moving down
            return (-ring, ring - 1 - position_in_side)
        else:  # Bottom side, moving right
            return (-ring + 1 + position_in_side, -ring)
    
    def vortex_math_reduce(self, n: int) -> int:
        """
        Vortex mathematics - reduce number to single digit (base 9)
        Tesla's 3-6-9 pattern
        """
        if n == 0:
            return 0
        reduced = n % 9
        return 9 if reduced == 0 else reduced
    
    def tesla_369_pattern(self, sequence: List[int]) -> List[int]:
        """
        Apply Tesla's 3-6-9 pattern to a sequence
        Identifies energy flow patterns
        """
        return [self.vortex_math_reduce(x) for x in sequence]
    
    def tesseract_projection(self, point_4d: np.ndarray) -> np.ndarray:
        """
        Project 4D tesseract point to 3D space
        Useful for high-dimensional feature space optimization
        """
        if len(point_4d) != 4:
            raise ValueError("Input must be 4-dimensional")
        
        # Simple orthographic projection (drop 4th dimension with scaling)
        w = point_4d[3]
        scale = 1 / (2 - w) if w != 2 else 1
        
        return point_4d[:3] * scale
    
    def tesseract_vertices(self) -> np.ndarray:
        """Generate all 16 vertices of a tesseract"""
        vertices = []
        for i in range(16):
            vertex = [
                1 if (i & 1) else -1,
                1 if (i & 2) else -1,
                1 if (i & 4) else -1,
                1 if (i & 8) else -1
            ]
            vertices.append(vertex)
        return np.array(vertices)
    
    def golden_section_search(self, f, a: float, b: float, tol: float = 1e-5) -> float:
        """
        Golden section search for function optimization
        Finds minimum of unimodal function f in interval [a, b]
        """
        inv_phi = 1 / self.PHI
        inv_phi2 = 1 / (self.PHI ** 2)
        
        h = b - a
        if h <= tol:
            return (a + b) / 2
        
        # Required steps
        n = int(math.ceil(math.log(tol / h) / math.log(inv_phi)))
        
        c = a + inv_phi2 * h
        d = a + inv_phi * h
        fc = f(c)
        fd = f(d)
        
        for _ in range(n - 1):
            if fc < fd:
                b = d
                d = c
                fd = fc
                h = inv_phi * h
                c = a + inv_phi2 * h
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                h = inv_phi * h
                d = a + inv_phi * h
                fd = f(d)
        
        return (a + b) / 2 if fc < fd else (c + d) / 2
    
    def optimize_layer_sizes(self, input_size: int, output_size: int, num_layers: int) -> List[int]:
        """
        Calculate optimal neural network layer sizes using golden ratio
        Creates harmonious scaling between input and output
        """
        if num_layers <= 2:
            return [input_size, output_size]
        
        sizes = [input_size]
        ratio = (output_size / input_size) ** (1 / (num_layers - 1))
        
        # Apply golden ratio adjustment
        phi_adjusted_ratio = ratio * (self.PHI / 2)
        
        for i in range(1, num_layers - 1):
            size = int(input_size * (phi_adjusted_ratio ** i))
            sizes.append(size)
        
        sizes.append(output_size)
        return sizes
    
    def hexagonal_grid_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get neighbors in hexagonal grid (Flower of Life pattern)
        More efficient than square grid for certain optimizations
        """
        if y % 2 == 0:
            return [
                (x, y - 1), (x + 1, y - 1),
                (x - 1, y), (x + 1, y),
                (x, y + 1), (x + 1, y + 1)
            ]
        else:
            return [
                (x - 1, y - 1), (x, y - 1),
                (x - 1, y), (x + 1, y),
                (x - 1, y + 1), (x, y + 1)
            ]
    
    def sacred_geometry_hash(self, data: bytes, table_size: int) -> int:
        """
        Combined sacred geometry hashing using multiple principles
        """
        # Convert bytes to integer
        value = int.from_bytes(data, byteorder='big')
        
        # Apply Fibonacci hashing
        fib_hash = self.fibonacci_hash(value, table_size)
        
        # Apply vortex math reduction
        vortex = self.vortex_math_reduce(value)
        
        # Combine with golden ratio
        combined = int((fib_hash * self.PHI + vortex) % table_size)
        
        return combined
    
    def chaos_temperature(self, vitality_score: float) -> float:
        """
        Calculate chaos temperature using sacred geometry
        Higher vitality = lower chaos (more stability)
        """
        # Use golden ratio to balance chaos and order
        base_chaos = 1 / self.PHI  # ≈ 0.618
        
        # Inverse relationship with vitality
        temperature = base_chaos * (1 - (vitality_score / 10))
        
        # Apply Tesla 3-6-9 modulation
        tesla_mod = self.vortex_math_reduce(int(vitality_score * 100)) / 9
        
        return temperature * (1 + tesla_mod * 0.1)
    
    def get_optimization_constants(self) -> Dict[str, Any]:
        """Return all sacred geometry constants for system-wide use"""
        return {
            "phi": self.PHI,
            "pi": self.PI,
            "tesla_numbers": self.TESLA_NUMBERS,
            "metatron_vertices": self.metatron_vertices.tolist(),
            "flower_of_life_centers": self.flower_of_life_centers.tolist(),
            "fibonacci_sequence_20": self.fibonacci_sequence(20),
            "platonic_solids": {
                "tetrahedron": self.platonic_solid_vertices("tetrahedron").tolist(),
                "cube": self.platonic_solid_vertices("cube").tolist(),
                "octahedron": self.platonic_solid_vertices("octahedron").tolist(),
                "dodecahedron": self.platonic_solid_vertices("dodecahedron").tolist(),
                "icosahedron": self.platonic_solid_vertices("icosahedron").tolist()
            }
        }


# Global instance for easy access
sacred_optimizer = SacredGeometryOptimizer()
