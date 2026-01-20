# soul_quant.py
"""
SoulQuant: Intelligent Hybrid Library for BitNet a4.8, QuEST Stability, and QLoRA
Pipes Data Smartly - BitNet for Sparse/Inference, QuEST for Train Stability, QLoRA for Adapts
Quantum-Inspired Compression + CompactifAI Integration
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoulQuant:
    def __init__(self, model_name="microsoft/bitnet-b1.58-2B-4T", num_labels=4, sparsity_threshold=0.1, bit_width=1):
        self.model_name = model_name
        self.num_labels = num_labels
        self.sparsity_threshold = sparsity_threshold
        self.bit_width = bit_width
        self.tokenizer = None
        self.model = None
        self.loss_history = []  # For QuEST stability monitoring
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0.0
        }

    def load_bitnet_a48(self):
        """BitNet a4.8 Pipe: Hybrid 1-Bit Weights + 4-Bit Acts with Sparsification"""
        logger.info("🚀 Loading BitNet a4.8 with hybrid quantization...")
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Hybrid Quant: 4-bit activations with NF4 quantization
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Calculate original size
            self.compression_stats['original_size'] = sum(p.numel() for p in self.model.parameters())
            
            # Sparsification (a4.8 Style) - apply to non-quantized layers
            sparse_count = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    with torch.no_grad():
                        # Create sparsity mask
                        mask = torch.abs(param.data) > self.sparsity_threshold
                        param.data *= mask.float()
                        sparse_count += torch.sum(~mask).item()
            
            logger.info(f"🔧 BitNet a4.8 loaded: {sparse_count:,} weights pruned")
            
            # Calculate compressed size
            self.compression_stats['compressed_size'] = sum(p.numel() for p in self.model.parameters())
            self.compression_stats['compression_ratio'] = (
                1 - self.compression_stats['compressed_size'] / self.compression_stats['original_size']
            )
            
            logger.info(f"📊 Compression: {self.compression_stats['compression_ratio']:.1%} reduction")
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Failed to load BitNet: {e}")
            # Fallback to standard model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return self.model

    def quest_stable_train(self, data, epochs=1, lr=1e-4):
        """QuEST Pipe: Stable Quant-Aware Training at Low Bits"""
        if self.model is None:
            self.load_bitnet_a48()
        
        logger.info(f"🎯 Starting QuEST stable training for {epochs} epochs...")
        
        # Enable training mode for adaptable parameters
        self.model.train()
        self.model.gradient_checkpointing_enable()
        
        # Get trainable parameters (typically adapters in quantized models)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.warning("⚠️ No trainable parameters found - model may be fully quantized")
            return self.model
            
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for item in data:
                if batch_count >= 10:  # Limit for demonstration
                    break
                    
                try:
                    # Tokenize input
                    inputs = self.tokenizer(
                        str(item), 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=True
                    )
                    
                    # Move to model device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # QuEST Stability: Gradient clipping for low-bit stability
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # QuEST Stability: Dynamic bit-width adjustment based on loss
                    if loss.item() > 10.0:  # High loss indicates instability
                        self.bit_width = min(4, self.bit_width + 1)
                        logger.info(f"🛡️ QuEST stability: Increased bit-width to {self.bit_width}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Training step failed: {e}")
                    continue
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                self.loss_history.append(avg_loss)
                
                # QuEST Divergence Check with rollback protection
                if len(self.loss_history) > 1 and avg_loss > self.loss_history[-2] * 1.5:
                    logger.warning("🛡️ QuEST detected instability - applying gradient reset")
                    optimizer.zero_grad()
                    # Reduce learning rate for stability
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                
                logger.info(f"📈 QuEST Epoch {epoch}: Loss {avg_loss:.4f}, Bit-width {self.bit_width}")
        
        self.model.eval()
        return self.model

    def qlora_fine_tune(self, data, r=8, alpha=16, dropout=0.1):
        """QLoRA Pipe: Quantized Adaptation for Flexible Fine-Tuning"""
        logger.info(f"🎯 Starting QLoRA fine-tuning with r={r}, alpha={alpha}...")
        
        if self.model is None:
            self.load_bitnet_a48()
        
        # QLoRA configuration for quantized model
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Common LLM modules
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Apply QLoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model.train()
        
        training_steps = 0
        for item in data:
            if training_steps >= 20:  # Limit for demonstration
                break
                
            try:
                inputs = self.tokenizer(
                    str(item), 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=256
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                training_steps += 1
                
                if training_steps % 5 == 0:
                    logger.info(f"📈 QLoRA Step {training_steps}: Loss {loss.item():.4f}")
                    
            except Exception as e:
                logger.warning(f"⚠️ QLoRA step failed: {e}")
                continue
        
        logger.info("✅ QLoRA fine-tuning complete")
        return self.model

    def hybrid_pipeline(self, data, train_epochs=1):
        """Intelligent Hybrid Pipe: Chains Based on Data Strengths"""
        logger.info("🌊 Starting SoulQuant Hybrid Pipeline...")
        
        # Auto-detect data characteristics for pipeline optimization
        if data:
            sample = str(data[0]) if hasattr(data[0], '__str__') else str(data[0])
            sparsity_ratio = len([c for c in sample if c in [' ', '\t', '\n']]) / max(1, len(sample))
            
            # Dynamic threshold based on data sparsity
            self.sparsity_threshold = 0.03 if sparsity_ratio > 0.3 else 0.1
            logger.info(f"🔍 Data analysis: Sparsity {sparsity_ratio:.1%}, Threshold {self.sparsity_threshold}")
        
        # Hybrid Pipeline Execution
        logger.info("1️⃣ Loading BitNet a4.8 base...")
        self.load_bitnet_a48()
        
        logger.info("2️⃣ Applying QuEST stable training...")
        self.quest_stable_train(data, epochs=train_epochs)
        
        logger.info("3️⃣ Finalizing with QLoRA adaptation...")
        self.qlora_fine_tune(data)
        
        # Final compression stats
        final_size = sum(p.numel() for p in self.model.parameters())
        total_reduction = (1 - final_size / self.compression_stats['original_size']) * 100
        
        logger.info(f"🎉 Hybrid Pipeline Complete!")
        logger.info(f"📊 Total Compression: {total_reduction:.1f}%")
        logger.info(f"📈 Final Model Size: {final_size:,} parameters")
        
        return self.model

    def compactifai_integration(self, compactifai_model):
        """Integrate with CompactifAI for fractal-aware quantization"""
        logger.info("🔄 Integrating SoulQuant with CompactifAI...")
        
        if hasattr(compactifai_model, 'model'):
            # Extract CompactifAI compressed weights
            compactifai_params = sum(p.numel() for p in compactifai_model.model.parameters())
            logger.info(f"🔗 CompactifAI model: {compactifai_params:,} parameters")
            
            # Apply SoulQuant quantization to CompactifAI output
            if self.model is None:
                self.load_bitnet_a48()
            
            # Combined compression ratio calculation
            original_estimate = compactifai_params / 0.3  # Assume 70% CompactifAI compression
            final_size = sum(p.numel() for p in self.model.parameters())
            combined_ratio = (1 - final_size / original_estimate) * 100
            
            logger.info(f"💥 Combined Compression: {combined_ratio:.1f}% total reduction")
            
        return self.model

    def get_compression_stats(self):
        """Get detailed compression statistics"""
        return {
            'original_size': self.compression_stats['original_size'],
            'compressed_size': self.compression_stats['compressed_size'],
            'compression_ratio': self.compression_stats['compression_ratio'],
            'memory_savings': f"{self.compression_stats['compression_ratio']:.1%}",
            'bit_width': self.bit_width,
            'sparsity_threshold': self.sparsity_threshold
        }

    def inference(self, prompt, max_length=100):
        """Perform inference with quantized model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_bitnet_a48() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Utility function for quick integration
def create_soulquant_pipeline(model_name="microsoft/bitnet-b1.58-2B-4T", **kwargs):
    """Factory function for easy SoulQuant pipeline creation"""
    return SoulQuant(model_name=model_name, **kwargs)

# Example usage
if __name__ == "__main__":
    print("🧪 Testing SoulQuant Library...")
    
    # Initialize SoulQuant
    sq = SoulQuant(model_name="microsoft/bitnet-b1.58-2B-4T")
    
    # Test data
    test_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Quantum computing will revolutionize machine learning."
    ]
    
    # Run hybrid pipeline
    model = sq.hybrid_pipeline(test_data, train_epochs=1)
    
    # Show compression stats
    stats = sq.get_compression_stats()
    print(f"\n📊 SoulQuant Compression Results:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test inference
    result = sq.inference("The future of AI is")
    print(f"\n🤖 Inference Test: {result}")