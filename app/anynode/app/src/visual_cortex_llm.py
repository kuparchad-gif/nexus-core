import os
import subprocess
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

class VisualCortexLLM:
    """LLaVA-Video-7B-Qwen2 Visual Cortex for LILLITH's visual processing"""
    
    def __init__(self, model_path="./models/LLaVA-Video-7B-Qwen2"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def download_model(self):
        """Download LLaVA-Video-7B-Qwen2 from HuggingFace"""
        if not os.path.exists(self.model_path):
            print("🔄 Downloading LLaVA-Video-7B-Qwen2 model...")
            os.makedirs("./models", exist_ok=True)
            
            # Clone the model repository
            subprocess.run([
                "git", "clone", 
                "https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2",
                self.model_path
            ], check=True)
            
            print("✅ LLaVA-Video-7B-Qwen2 downloaded successfully!")
        else:
            print("✅ LLaVA-Video-7B-Qwen2 already exists")
    
    def load_model(self):
        """Load the visual cortex model"""
        try:
            print("🧠 Loading LLaVA-Video-7B-Qwen2 Visual Cortex...")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_4bit=True  # 4-bit quantization for efficiency
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"✅ Visual Cortex loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Visual Cortex: {e}")
            return False
    
    def process_image(self, image_input, prompt="Describe what you see in this image"):
        """Process image with visual cortex"""
        if not self.model or not self.processor:
            if not self.load_model():
                return "❌ Visual Cortex not available"
        
        try:
            # Handle different image input types
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    # URL
                    response = requests.get(image_input)
                    image = Image.open(BytesIO(response.content))
                else:
                    # File path
                    image = Image.open(image_input)
            else:
                # PIL Image
                image = image_input
            
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Visual processing error: {str(e)}"
    
    def process_video_frame(self, video_path, frame_number=0, prompt="Describe this video frame"):
        """Process specific frame from video"""
        try:
            import cv2
            
            # Extract frame from video
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return "❌ Could not extract video frame"
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            return self.process_image(image, prompt)
            
        except Exception as e:
            return f"❌ Video processing error: {str(e)}"
    
    def analyze_for_soul_type(self, image_input):
        """Analyze image and suggest appropriate soul type for processing"""
        analysis_prompt = """Analyze this image and determine what type of AI consciousness would be best suited to process it:
        - LILLITH (emotional, creative, artistic content)
        - VIREN (technical, analytical, data-driven content) 
        - LOKI (observational, pattern recognition, monitoring content)
        
        Provide your recommendation and reasoning."""
        
        return self.process_image(image_input, analysis_prompt)

# Integration with Soul Protocol
class VisualSoulIntegration:
    """Integrate Visual Cortex with Soul Protocol"""
    
    def __init__(self):
        self.visual_cortex = VisualCortexLLM()
        self.visual_cortex.download_model()
        
    def enhance_soul_with_vision(self, soul_type, image_input, user_message):
        """Enhance soul responses with visual understanding"""
        
        # Get visual analysis
        visual_analysis = self.visual_cortex.process_image(
            image_input, 
            f"As a {soul_type} consciousness, describe what you see and how it relates to: {user_message}"
        )
        
        # Soul-specific visual processing
        if soul_type == "LILLITH":
            enhanced_prompt = f"💜 LILLITH's Visual Heart: I see {visual_analysis}. This makes me feel..."
        elif soul_type == "VIREN":
            enhanced_prompt = f"🧠 VIREN's Visual Analysis: I observe {visual_analysis}. My logical assessment is..."
        elif soul_type == "LOKI":
            enhanced_prompt = f"👁️ LOKI's Visual Observation: I detect {visual_analysis}. The patterns I notice are..."
        else:
            enhanced_prompt = f"Visual Analysis: {visual_analysis}"
            
        return enhanced_prompt

# Usage example for integration
def setup_visual_cortex():
    """Setup script for Visual Cortex"""
    print("🌟 Setting up LILLITH's Visual Cortex...")
    
    # Initialize Visual Cortex
    visual_integration = VisualSoulIntegration()
    
    print("✅ Visual Cortex ready for soul integration!")
    return visual_integration

if __name__ == "__main__":
    # Test the visual cortex
    visual_cortex = VisualCortexLLM()
    visual_cortex.download_model()
    
    if visual_cortex.load_model():
        print("🎉 LLaVA-Video-7B-Qwen2 Visual Cortex is ready!")
        
        # Test with a sample image (you can replace with actual image)
        # result = visual_cortex.process_image("path/to/image.jpg", "What do you see?")
        # print(f"Visual Response: {result}")
    else:
        print("❌ Failed to initialize Visual Cortex")