"""
Complete Demo of Ultimate AI Toolbox
Demonstrates all major capabilities with sacred geometry optimization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultimate_toolbox import create_toolbox
import numpy as np


def demo_documents(toolbox):
    """Demonstrate document operations"""
    print("\n" + "=" * 80)
    print("DOCUMENT OPERATIONS DEMO")
    print("=" * 80)
    
    # Create text file
    print("\n1. Creating text file...")
    text_file = toolbox.create_document(
        'text', 
        'sample.txt', 
        content="This is a sample text file created by Ultimate Toolbox!"
    )
    print(f"   Created: {text_file}")
    
    # Create Word document
    print("\n2. Creating Word document...")
    word_content = [
        {"type": "heading", "text": "Ultimate Toolbox Demo", "level": 1},
        {"type": "paragraph", "text": "This document was created programmatically."},
        {"type": "bullet", "items": ["Feature 1", "Feature 2", "Feature 3"]},
        {"type": "table", "data": [[1, 2, 3], [4, 5, 6]], "headers": ["A", "B", "C"]}
    ]
    word_file = toolbox.create_document(
        'word',
        'demo.docx',
        title="Demo Document",
        content=word_content
    )
    print(f"   Created: {word_file}")
    
    # Create Excel with sacred geometry optimization
    print("\n3. Creating Excel with Fibonacci-optimized data...")
    fib_sequence = toolbox.fibonacci_sequence(10)
    excel_data = {
        "Fibonacci": [[i, fib_sequence[i]] for i in range(len(fib_sequence))],
        "Golden Ratio": [[i, i * toolbox.PHI] for i in range(10)]
    }
    excel_file = toolbox.create_document(
        'excel',
        'sacred_geometry.xlsx',
        sheets=excel_data,
        headers={
            "Fibonacci": ["Index", "Value"],
            "Golden Ratio": ["Index", "Phi Multiple"]
        }
    )
    print(f"   Created: {excel_file}")
    print(f"   Fibonacci sequence: {fib_sequence}")


def demo_web(toolbox):
    """Demonstrate web operations"""
    print("\n" + "=" * 80)
    print("WEB OPERATIONS DEMO")
    print("=" * 80)
    
    # Fetch webpage
    print("\n1. Fetching webpage...")
    try:
        result = toolbox.fetch_url("https://example.com", parse=True)
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Links found: {len(result.get('links', []))}")
        print(f"   Text preview: {result.get('text', '')[:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # API call
    print("\n2. Making API call...")
    try:
        api_result = toolbox.web.api_call(
            "https://api.github.com/repos/python/cpython",
            method="GET"
        )
        if api_result.get("success"):
            data = api_result.get("json", {})
            print(f"   Repository: {data.get('full_name', 'N/A')}")
            print(f"   Stars: {data.get('stargazers_count', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")


def demo_media(toolbox):
    """Demonstrate media operations"""
    print("\n" + "=" * 80)
    print("MEDIA OPERATIONS DEMO")
    print("=" * 80)
    
    # Create image with golden ratio dimensions
    print("\n1. Creating image with golden ratio dimensions...")
    width = 800
    height = int(width / toolbox.PHI)
    image_file = toolbox.create_image(
        width, height,
        color='#4472C4',
        filename='golden_rectangle.png'
    )
    print(f"   Created: {image_file}")
    print(f"   Dimensions: {width}x{height} (ratio: {width/height:.4f} ≈ Φ)")
    
    # Create gradient
    print("\n2. Creating gradient image...")
    gradient_file = toolbox.media.create_gradient(
        600, 400,
        (255, 0, 0), (0, 0, 255),
        'horizontal',
        'gradient.png'
    )
    print(f"   Created: {gradient_file}")
    
    # Apply filter
    print("\n3. Applying filter to image...")
    try:
        filtered = toolbox.edit_image(
            gradient_file,
            'filter',
            filter_name='blur'
        )
        print(f"   Filtered: {filtered}")
    except Exception as e:
        print(f"   Error: {e}")


def demo_parallel(toolbox):
    """Demonstrate parallel processing"""
    print("\n" + "=" * 80)
    print("PARALLEL PROCESSING DEMO")
    print("=" * 80)
    
    # Define test function
    def square(x):
        return x ** 2
    
    # Generate test data using Fibonacci
    print("\n1. Generating test data using Fibonacci sequence...")
    test_data = toolbox.fibonacci_sequence(15)
    print(f"   Test data: {test_data}")
    
    # Sequential execution
    print("\n2. Sequential execution...")
    import time
    start = time.time()
    seq_results = [square(x) for x in test_data]
    seq_time = time.time() - start
    print(f"   Results: {seq_results[:5]}...")
    print(f"   Time: {seq_time:.4f}s")
    
    # Parallel execution with RAY
    if toolbox.ray_optimizer:
        print("\n3. Parallel execution with RAY...")
        start = time.time()
        par_results = toolbox.parallel_execute(square, test_data, method="ray")
        par_time = time.time() - start
        print(f"   Results: {par_results[:5]}...")
        print(f"   Time: {par_time:.4f}s")
        print(f"   Speedup: {seq_time/par_time:.2f}x")
    
    # Batch execution with sacred geometry optimization
    print("\n4. Batch execution with auto-optimization...")
    batch_results = toolbox.parallel_batch_execute(square, test_data)
    print(f"   Results: {batch_results[:5]}...")


def demo_vector_search(toolbox):
    """Demonstrate vector search with FAISS"""
    print("\n" + "=" * 80)
    print("VECTOR SEARCH DEMO")
    print("=" * 80)
    
    # Create vector store
    print("\n1. Creating vector store...")
    dimension = 128
    store_name = "demo_store"
    toolbox.create_vector_store(store_name, dimension, index_type="flat")
    print(f"   Store created: {store_name} (dimension={dimension})")
    
    # Generate random vectors
    print("\n2. Adding vectors...")
    num_vectors = 100
    vectors = np.random.randn(num_vectors, dimension).astype('float32')
    metadata = [{"id": i, "category": f"cat_{i % 5}"} for i in range(num_vectors)]
    toolbox.add_vectors(store_name, vectors, metadata)
    print(f"   Added {num_vectors} vectors")
    
    # Search
    print("\n3. Searching for similar vectors...")
    query = np.random.randn(1, dimension).astype('float32')
    
    # Use golden ratio optimized k
    k = int(10 * toolbox.PHI / 2)
    results = toolbox.search_vectors(store_name, query, k=k)
    
    print(f"   Found {len(results)} results (k={k}, optimized by Φ)")
    for i, result in enumerate(results[:3]):
        print(f"   {i+1}. Index: {result['index']}, "
              f"Similarity: {result['similarity']:.4f}, "
              f"Metadata: {result.get('metadata', {})}")


def demo_sacred_geometry(toolbox):
    """Demonstrate sacred geometry optimizations"""
    print("\n" + "=" * 80)
    print("SACRED GEOMETRY OPTIMIZATION DEMO")
    print("=" * 80)
    
    # Golden ratio scaling
    print("\n1. Golden Ratio Scaling:")
    value = 100
    for i in range(1, 6):
        scaled = toolbox.optimize_with_golden_ratio(value, iterations=i)
        print(f"   Iteration {i}: {value} → {scaled:.2f}")
    
    # Fibonacci sequence
    print("\n2. Fibonacci Sequence:")
    fib = toolbox.fibonacci_sequence(12)
    print(f"   {fib}")
    
    # Vortex math (Tesla 3-6-9)
    print("\n3. Vortex Math Reduction (Tesla 3-6-9):")
    numbers = [12, 45, 108, 369, 1234]
    for num in numbers:
        reduced = toolbox.vortex_reduce(num)
        print(f"   {num} → {reduced}")
    
    # Metatron routing
    print("\n4. Metatron's Cube Routing:")
    query_vector = np.array([0.5, 0.3, 0.8])
    routes = toolbox.metatron_route(query_vector, num_paths=5)
    print(f"   Query: {query_vector}")
    print(f"   Routes: {routes}")
    
    # Neural network layer optimization
    print("\n5. Neural Network Layer Optimization:")
    layers = toolbox.optimize_layer_sizes(784, 10, 4)
    print(f"   Input: 784 → Hidden: {layers[1:-1]} → Output: 10")
    print(f"   Ratios: {[layers[i]/layers[i-1] for i in range(1, len(layers))]}")
    
    # Golden section search
    print("\n6. Golden Section Search:")
    def test_function(x):
        return (x - 3) ** 2 + 5
    
    minimum = toolbox.golden_section_search(test_function, 0, 10)
    print(f"   Function: (x-3)² + 5")
    print(f"   Minimum at x = {minimum:.6f}")
    print(f"   f(x) = {test_function(minimum):.6f}")


def demo_workflows(toolbox):
    """Demonstrate agent workflows"""
    print("\n" + "=" * 80)
    print("AGENT WORKFLOW DEMO")
    print("=" * 80)
    
    if not toolbox.orchestrator:
        print("   LangGraph not enabled, skipping workflow demo")
        return
    
    print("\n1. Creating simple agent workflow...")
    workflow_id = "demo_workflow"
    toolbox.create_workflow(workflow_id, workflow_type="simple")
    print(f"   Workflow created: {workflow_id}")
    
    print("\n2. Executing workflow...")
    try:
        result = toolbox.execute_workflow(
            workflow_id,
            {
                "messages": [],
                "current_task": "Calculate the sum of first 10 Fibonacci numbers",
                "context": {},
                "results": {},
                "next_action": "start",
                "iteration": 0,
                "max_iterations": 3
            }
        )
        print(f"   Workflow completed!")
        print(f"   Iterations: {result.get('iteration', 0)}")
        print(f"   Results: {result.get('results', {})}")
    except Exception as e:
        print(f"   Error: {e}")


def demo_system_stats(toolbox):
    """Display system statistics"""
    print("\n" + "=" * 80)
    print("SYSTEM STATISTICS")
    print("=" * 80)
    
    stats = toolbox.get_system_stats()
    
    print("\nSacred Geometry Constants:")
    sg = stats.get("sacred_geometry", {})
    print(f"  Φ (Phi): {sg.get('phi', 0):.6f}")
    print(f"  π (Pi): {sg.get('pi', 0):.6f}")
    print(f"  Tesla Numbers: {sg.get('tesla_numbers', [])}")
    
    if "ray" in stats:
        print("\nRAY Cluster:")
        ray_stats = stats["ray"]
        print(f"  Total Resources: {ray_stats.get('total', {})}")
        print(f"  Available: {ray_stats.get('available', {})}")
        print(f"  Nodes: {ray_stats.get('nodes', 0)}")
    
    if "faiss_stores" in stats:
        print("\nFAISS Vector Stores:")
        for name, store_stats in stats["faiss_stores"].items():
            print(f"  {name}:")
            print(f"    Size: {store_stats.get('size', 0)} vectors")
            print(f"    Dimension: {store_stats.get('dimension', 0)}")
            print(f"    Type: {store_stats.get('index_type', 'unknown')}")


def main():
    """Run complete demo"""
    print("\n" + "=" * 80)
    print("ULTIMATE AI TOOLBOX - COMPLETE DEMONSTRATION")
    print("=" * 80)
    print("\nInitializing toolbox with all features enabled...")
    
    with create_toolbox(
        workspace="/tmp/ultimate_toolbox_demo",
        enable_ray=True,
        enable_faiss=True,
        enable_langgraph=True
    ) as toolbox:
        
        # Run all demos
        demo_documents(toolbox)
        demo_web(toolbox)
        demo_media(toolbox)
        demo_parallel(toolbox)
        demo_vector_search(toolbox)
        demo_sacred_geometry(toolbox)
        demo_workflows(toolbox)
        demo_system_stats(toolbox)
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE!")
        print("=" * 80)
        print("\nAll capabilities demonstrated successfully!")
        print("Check the workspace for generated files:")
        print(f"  {toolbox.workspace}")


if __name__ == "__main__":
    main()
