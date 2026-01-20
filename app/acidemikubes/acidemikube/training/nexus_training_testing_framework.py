# nexus_training_testing_framework.py
"""
🧪 NEXUS TRAINING & TESTING FRAMEWORK
Benchmark original vs compressed models - prove our superiority
"""

import torch
import time
import psutil
import GPUtil
from transformers import AutoModelForCausalLM, AutoTokenizer
from soul_quant import SoulQuant
from real_compactifi_train import TrueCompactifAI

class NexusTestingFramework:
    """Comprehensive testing of compressed vs original models"""
    
    def __init__(self):
        self.test_datasets = self.prepare_test_datasets()
        self.metrics_tracker = {}
        
    def prepare_test_datasets(self):
        """Diverse test datasets for comprehensive evaluation"""
        return {
            'code_generation': [
                "Write a Python function to calculate fibonacci numbers",
                "Create a React component for a login form",
                "Implement binary search in JavaScript"
            ],
            'reasoning_tasks': [
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "You have two ropes that each take exactly 60 minutes to burn, but they burn at inconsistent rates. How do you measure 45 minutes?"
            ],
            'creative_writing': [
                "Write a short story about a robot who discovers emotions",
                "Create a poem about the changing seasons",
                "Write a dialogue between two ancient philosophers"
            ],
            'technical_questions': [
                "Explain quantum entanglement in simple terms",
                "What are the differences between TCP and UDP?",
                "How does a transformer architecture work in AI models?"
            ]
        }
    
    def measure_memory_usage(self):
        """Measure current memory usage"""
        cpu_memory = psutil.virtual_memory()
        gpu_memory = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_memory.append({
                    'id': gpu.id,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal
                })
        except:
            gpu_memory = [{'error': 'GPU not available'}]
        
        return {
            'cpu_used_gb': cpu_memory.used / (1024**3),
            'cpu_total_gb': cpu_memory.total / (1024**3),
            'gpu_memory': gpu_memory
        }
    
    def benchmark_model(self, model, tokenizer, model_name):
        """Comprehensive benchmarking of a single model"""
        print(f"🧪 BENCHMARKING: {model_name}")
        
        benchmarks = {
            'model_name': model_name,
            'memory_usage': self.measure_memory_usage(),
            'inference_speed': {},
            'quality_scores': {},
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'test_results': {}
        }
        
        # Test inference speed across different tasks
        for task_type, prompts in self.test_datasets.items():
            print(f"   ⚡ Testing {task_type}...")
            
            task_times = []
            task_responses = []
            
            for prompt in prompts[:2]:  # Test first 2 prompts per category
                start_time = time.time()
                
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=200,
                            num_return_sequences=1,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    end_time = time.time()
                    
                    task_times.append(end_time - start_time)
                    task_responses.append(response)
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    task_times.append(float('inf'))
                    task_responses.append(f"ERROR: {e}")
            
            benchmarks['inference_speed'][task_type] = {
                'avg_time': sum(task_times) / len(task_times) if task_times else float('inf'),
                'total_time': sum(task_times)
            }
            
            benchmarks['test_results'][task_type] = task_responses
        
        return benchmarks
    
    def run_comparison_test(self, original_model_name="deepseek-ai/deepseek-llm-7b-base"):
        """Compare original model vs our compressed versions"""
        print("=" * 70)
        print("🧪 NEXUS COMPRESSION COMPARISON TEST")
        print("⚡ ORIGINAL vs COMPACTIFAI vs SOULQUANT vs HYBRID")
        print("=" * 70)
        
        results = {}
        
        # Test 1: Original Model Baseline
        print("\n🎯 TEST 1: ORIGINAL MODEL BASELINE")
        original_model = AutoModelForCausalLM.from_pretrained(original_model_name)
        original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
        results['original'] = self.benchmark_model(original_model, original_tokenizer, "Original")
        
        # Test 2: CompactifAI Compressed
        print("\n🎯 TEST 2: COMPACTIFAI COMPRESSION")
        compactifai = TrueCompactifAI(model_name=original_model_name)
        compact_model = compactifai.compress_model()
        results['compactifai'] = self.benchmark_model(compact_model, original_tokenizer, "CompactifAI")
        
        # Test 3: SoulQuant Compressed  
        print("\n🎯 TEST 3: SOULQUANT COMPRESSION")
        soulquant = SoulQuant(model_name=original_model_name)
        soulquant_model = soulquant.hybrid_pipeline([])  # No training data for base compression
        results['soulquant'] = self.benchmark_model(soulquant_model, original_tokenizer, "SoulQuant")
        
        # Test 4: Hybrid Compression (Both)
        print("\n🎯 TEST 4: HYBRID COMPRESSION")
        # First CompactifAI, then SoulQuant
        hybrid_model = soulquant.compactifai_integration(compactifai)
        results['hybrid'] = self.benchmark_model(hybrid_model, original_tokenizer, "Hybrid")
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate comprehensive comparison report"""
        print("\n" + "📊" * 20)
        print("COMPREHENSIVE TEST RESULTS")
        print("📊" * 20)
        
        report = {
            'compression_ratios': {},
            'speed_improvements': {},
            'memory_savings': {},
            'quality_assessment': {}
        }
        
        original_params = results['original']['parameter_count']
        
        for method, result in results.items():
            if method != 'original':
                # Compression ratio
                compression_ratio = 1 - (result['parameter_count'] / original_params)
                report['compression_ratios'][method] = compression_ratio
                
                # Speed improvements (lower is better)
                original_speed = results['original']['inference_speed']['code_generation']['avg_time']
                method_speed = result['inference_speed']['code_generation']['avg_time']
                speed_improvement = original_speed / method_speed if method_speed > 0 else 0
                report['speed_improvements'][method] = speed_improvement
                
                # Memory savings
                original_memory = results['original']['memory_usage']['cpu_used_gb']
                method_memory = result['memory_usage']['cpu_used_gb']
                memory_savings = 1 - (method_memory / original_memory) if original_memory > 0 else 0
                report['memory_savings'][method] = memory_savings
        
        # Print summary
        print(f"\n🎯 COMPRESSION PERFORMANCE SUMMARY:")
        for method in ['compactifai', 'soulquant', 'hybrid']:
            print(f"   {method.upper():<12} | "
                  f"Compression: {report['compression_ratios'][method]:.1%} | "
                  f"Speed: {report['speed_improvements'][method]:.2f}x | "
                  f"Memory: {report['memory_savings'][method]:.1%}")
        
        return report

class AdvancedTestingScenarios:
    """More advanced testing scenarios"""
    
    def __init__(self):
        self.scenarios = self.define_test_scenarios()
    
    def define_test_scenarios(self):
        """Different real-world testing scenarios"""
        return {
            'edge_device_simulation': {
                'description': 'Test on resource-constrained environment',
                'constraints': {'max_memory_gb': 4, 'max_model_size_gb': 2},
                'metrics': ['inference_latency', 'memory_usage', 'power_consumption']
            },
            'multi_model_workload': {
                'description': 'Test running multiple compressed models simultaneously', 
                'constraints': {'total_memory_gb': 16, 'max_models': 10},
                'metrics': ['total_throughput', 'individual_performance', 'resource_sharing']
            },
            'progressive_compression': {
                'description': 'Test different compression levels',
                'levels': ['light', 'medium', 'aggressive', 'extreme'],
                'metrics': ['quality_loss', 'size_reduction', 'speed_impact']
            }
        }
    
    def run_edge_device_test(self, models_dict):
        """Simulate edge device constraints"""
        print("\n📱 EDGE DEVICE SIMULATION TEST")
        print("💾 Testing under resource constraints...")
        
        edge_results = {}
        for name, model_info in models_dict.items():
            model_size_gb = model_info['parameter_count'] * 4 / (1024**3)  # Approximate size in GB
            
            if model_size_gb <= 2:  # Edge device constraint
                print(f"   ✅ {name}: Fits in edge device ({model_size_gb:.2f} GB)")
                edge_results[name] = {
                    'fits': True,
                    'size_gb': model_size_gb,
                    'performance': 'testable'
                }
            else:
                print(f"   ❌ {name}: Too large for edge device ({model_size_gb:.2f} GB)")
                edge_results[name] = {
                    'fits': False, 
                    'size_gb': model_size_gb,
                    'performance': 'untestable'
                }
        
        return edge_results

# 🚀 COMPLETE TESTING PIPELINE
def run_complete_testing_pipeline():
    """Run the entire testing pipeline"""
    
    print("=" * 70)
    print("🚀 NEXUS COMPRESSION TECHNOLOGY VALIDATION")
    print("🧪 PROVING OUR SUPERIORITY THROUGH TESTING")
    print("=" * 70)
    
    # Initialize testing framework
    tester = NexusTestingFramework()
    advanced_tester = AdvancedTestingScenarios()
    
    # Run main comparison test
    print("🎯 Starting main compression comparison...")
    test_results = tester.run_comparison_test()
    
    # Generate comprehensive report
    print("\n📈 Generating test report...")
    comparison_report = tester.generate_comparison_report(test_results)
    
    # Run advanced scenarios
    print("\n🔬 Running advanced testing scenarios...")
    edge_results = advanced_tester.run_edge_device_test(test_results)
    
    # Final assessment
    print("\n" + "🎉" * 20)
    print("TESTING COMPLETE - RESULTS SUMMARY")
    print("🎉" * 20)
    
    best_method = max(comparison_report['compression_ratios'].items(), key=lambda x: x[1])
    print(f"🏆 BEST COMPRESSION: {best_method[0]} ({best_method[1]:.1%} reduction)")
    
    fastest_method = max(comparison_report['speed_improvements'].items(), key=lambda x: x[1])
    print(f"⚡ FASTEST INFERENCE: {fastest_method[0]} ({fastest_method[1]:.2f}x speedup)")
    
    # Determine if we achieved our goals
    hybrid_compression = comparison_report['compression_ratios'].get('hybrid', 0)
    if hybrid_compression >= 0.9:  # 90%+ compression
        print("✅ GOAL ACHIEVED: 90%+ compression demonstrated!")
    else:
        print("⚠️  GOAL PROGRESS: {:.1%} compression - room for improvement".format(hybrid_compression))
    
    return {
        'test_results': test_results,
        'comparison_report': comparison_report,
        'edge_results': edge_results
    }

# 🎯 QUICK START TESTING
def quick_test(model_path_or_name):
    """Quick test for a specific model"""
    print(f"⚡ QUICK TESTING: {model_path_or_name}")
    
    tester = NexusTestingFramework()
    
    try:
        # Try to load model
        model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
        
        # Run basic benchmark
        results = tester.benchmark_model(model, tokenizer, model_path_or_name)
        
        print(f"✅ Quick test completed for {model_path_or_name}")
        print(f"   Parameters: {results['parameter_count']:,}")
        print(f"   Memory: {results['memory_usage']['cpu_used_gb']:.2f} GB")
        
        return results
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return None

if __name__ == "__main__":
    # Run complete testing pipeline
    full_results = run_complete_testing_pipeline()
    
    print(f"\n🎯 Testing framework ready!")
    print(f"💡 Use quick_test('model_name') to test specific models")
    print(f"🚀 Or run_complete_testing_pipeline() for full comparison")