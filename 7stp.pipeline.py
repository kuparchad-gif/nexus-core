"""
Example usage of the Divine Pipeline module
"""

import numpy as np
from divine_pipeline import DivinePipeline, PipelineConfig

def example_1_basic_usage():
    """Basic usage example"""
    print("Example 1: Basic Usage")
    print("=" * 50)
    
    # Generate synthetic embeddings (simulating LLM output)
    np.random.seed(42)
    n_samples = 1000
    n_dims = 384  # Typical embedding dimension
    
    embeddings = np.random.randn(n_samples, n_dims) * 0.5
    labels = np.random.choice([0, 1, 2], n_samples)
    
    # Create pipeline with default config
    pipeline = DivinePipeline()
    
    # Apply pipeline
    enhanced_features = pipeline.fit_transform(embeddings, labels, verbose=True)
    
    print(f"\nOriginal embeddings shape: {embeddings.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    
    # Now use enhanced features for training
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        enhanced_features, labels, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy with divine features: {accuracy:.4f}")
    
    # Compare with raw embeddings
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    model_raw.fit(X_train_raw, y_train_raw)
    y_pred_raw = model_raw.predict(X_test_raw)
    accuracy_raw = accuracy_score(y_test_raw, y_pred_raw)
    
    print(f"Model accuracy with raw embeddings: {accuracy_raw:.4f}")
    print(f"Improvement: {(accuracy/accuracy_raw - 1)*100:+.2f}%")
    
    return pipeline

def example_2_custom_config():
    """Example with custom configuration"""
    print("\n\nExample 2: Custom Configuration")
    print("=" * 50)
    
    # Create custom configuration
    config = PipelineConfig(
        keep_ratio=0.9,  # Keep 90% of samples
        n_concepts=5,    # Use 5 divine concept anchors
        target_dim=100,  # Reduce to 100 dimensions
        n_clusters=8,    # 8 cosmic clusters
        include_flower_of_life=True,
        include_platonic_solids=False,  # Disable platonic solids
        compress_to_original=False,     # Don't compress to original dimension
    )
    
    # Create pipeline with custom config
    pipeline = DivinePipeline(config)
    
    # Generate data
    np.random.seed(123)
    embeddings = np.random.randn(500, 256)
    labels = np.random.choice([0, 1], 500)
    
    # Apply pipeline
    enhanced_features = pipeline.fit_transform(embeddings, labels, verbose=True)
    
    print(f"\nCustom pipeline configuration applied!")
    print(f"Feature info: {pipeline.get_feature_info()}")
    
    return pipeline

def example_3_integration_with_real_embeddings():
    """Example integrating with real sentence embeddings"""
    print("\n\nExample 3: Integration with Sentence Transformers")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load a real embedding model
        print("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Python is a powerful programming language for data science.",
            "Machine learning models require careful feature engineering.",
            "Natural language processing helps computers understand human language.",
            "Deep learning has revolutionized computer vision tasks.",
            "The weather today is sunny and warm.",
            "Data scientists use statistics to extract insights from data.",
            "Neural networks are inspired by the human brain.",
            "Feature selection improves model performance and interpretability."
        ]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = model.encode(texts)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Apply divine pipeline
        print("\nApplying Divine Pipeline...")
        pipeline = DivinePipeline()
        enhanced_features = pipeline.fit_transform(embeddings, verbose=True)
        
        print(f"\nEnhanced features ready for downstream tasks!")
        print(f"Sample features shape: {enhanced_features.shape}")
        
        # Show feature importance
        feature_info = pipeline.get_feature_info()
        print("\nFeature generation summary:")
        for trick, shape in feature_info['feature_shapes'].items():
            if trick != 'final':
                print(f"  {trick}: {shape[1]} features")
        
        return enhanced_features
        
    except ImportError:
        print("SentenceTransformers not installed. Install with: pip install sentence-transformers")
        return None

def example_4_save_and_load():
    """Example of saving and loading pipeline"""
    print("\n\nExample 4: Save and Load Pipeline")
    print("=" * 50)
    
    # Create and fit pipeline
    np.random.seed(42)
    embeddings = np.random.randn(100, 128)
    
    pipeline = DivinePipeline()
    enhanced_features = pipeline.fit_transform(embeddings, verbose=False)
    
    # Save pipeline
    pipeline.save('divine_pipeline.pkl')
    print("Pipeline saved to 'divine_pipeline.pkl'")
    
    # Load pipeline
    loaded_pipeline = DivinePipeline.load('divine_pipeline.pkl')
    print("Pipeline loaded successfully!")
    
    # Transform new data with loaded pipeline
    new_embeddings = np.random.randn(50, 128)
    new_features = loaded_pipeline.transform(new_embeddings, verbose=False)
    
    print(f"Transformed new data shape: {new_features.shape}")
    
    import os
    if os.path.exists('divine_pipeline.pkl'):
        os.remove('divine_pipeline.pkl')
        print("Cleanup: removed saved pipeline file")
    
    return loaded_pipeline

def example_5_batch_processing():
    """Example for large datasets with batch processing"""
    print("\n\nExample 5: Batch Processing for Large Datasets")
    print("=" * 50)
    
    # Generate large dataset
    n_samples = 10000
    n_dims = 384
    batch_size = 1000
    
    print(f"Generating {n_samples} samples...")
    all_embeddings = np.random.randn(n_samples, n_dims)
    all_labels = np.random.choice([0, 1], n_samples)
    
    # Initialize pipeline
    pipeline = DivinePipeline()
    
    # Process in batches
    all_enhanced_features = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_embeddings = all_embeddings[i:batch_end]
        batch_labels = all_labels[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}...")
        
        batch_features = pipeline.fit_transform(
            batch_embeddings, 
            batch_labels, 
            verbose=False
        )
        all_enhanced_features.append(batch_features)
    
    # Combine all batches
    enhanced_features = np.vstack(all_enhanced_features)
    
    print(f"\nBatch processing complete!")
    print(f"Original shape: {all_embeddings.shape}")
    print(f"Enhanced shape: {enhanced_features.shape}")
    
    return enhanced_features

if __name__ == "__main__":
    print("DIVINE PIPELINE EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_custom_config()
    example_3_integration_with_real_embeddings()
    example_4_save_and_load()
    example_5_batch_processing()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("\nUsage Summary:")
    print("1. Install: pip install numpy scipy scikit-learn")
    print("2. Import: from divine_pipeline import DivinePipeline, PipelineConfig")
    print("3. Create: pipeline = DivinePipeline()")
    print("4. Apply: enhanced_features = pipeline.fit_transform(embeddings)")
    print("5. Train: model.fit(enhanced_features, labels)")