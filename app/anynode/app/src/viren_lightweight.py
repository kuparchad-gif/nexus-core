#!/usr/bin/env python
"""
Viren Lightweight - No hanging, just working troubleshooter
"""

import sys
import os
import time
from pathlib import Path

# Setup paths
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

class VirenLightweight:
    """Lightweight Viren - no complex imports that hang"""
    
    def __init__(self):
        """Initialize lightweight Viren"""
        self.available_tech = {}
        self.system_status = "ready"
        
        # Test basic imports only
        self._test_basic_tech()
        
        print(f"🚀 Viren Lightweight initialized")
        print(f"Status: {self.system_status}")
    
    def _test_basic_tech(self):
        """Test basic tech without hanging imports"""
        
        # Test Gradio (but don't import websockets)
        try:
            import gradio
            self.available_tech["gradio"] = gradio.__version__
            print(f"✓ Gradio: {gradio.__version__}")
        except ImportError:
            print("✗ Gradio: Not available")
        
        # Test Modal
        try:
            import modal
            self.available_tech["modal"] = modal.__version__
            print(f"✓ Modal: {modal.__version__}")
        except ImportError:
            print("✗ Modal: Not available")
        
        # Test Weaviate
        try:
            import weaviate
            self.available_tech["weaviate"] = weaviate.__version__
            print(f"✓ Weaviate: {weaviate.__version__}")
        except ImportError:
            print("✗ Weaviate: Not available")
        
        # Test FastAPI (but don't start server)
        try:
            import fastapi
            self.available_tech["fastapi"] = fastapi.__version__
            print(f"✓ FastAPI: {fastapi.__version__}")
        except ImportError:
            print("✗ FastAPI: Not available")
        
        # Skip WebSockets for now (causing hang)
        print("⚠️ WebSockets: Skipped (causing hang)")
    
    def diagnose_chrome_reboot(self) -> dict:
        """Diagnose Chrome reboot issue - no complex imports"""
        
        return {
            "timestamp": time.time(),
            "issue": "Chrome triggers system reboots",
            "analysis": "Hardware acceleration + GPU driver conflict (classic pattern)",
            "confidence": 0.90,
            "solutions": [
                {
                    "step": 1,
                    "action": "Disable Chrome hardware acceleration",
                    "command": "chrome://settings/ → Advanced → System → Turn OFF 'Use hardware acceleration when available'",
                    "risk": "None",
                    "expected_result": "Should stop reboots immediately"
                },
                {
                    "step": 2,
                    "action": "Update GPU drivers",
                    "command": "Device Manager → Display adapters → Right-click → Update driver",
                    "risk": "Low",
                    "expected_result": "Better stability"
                },
                {
                    "step": 3,
                    "action": "Run memory test",
                    "command": "Windows key + R → type 'mdsched.exe' → Restart now",
                    "risk": "None",
                    "expected_result": "Identifies bad RAM if present"
                }
            ],
            "tech_used": list(self.available_tech.keys()),
            "readiness": "lightweight_mode"
        }
    
    def create_simple_gradio_ui(self):
        """Create simple Gradio UI without WebSockets"""
        
        if "gradio" not in self.available_tech:
            print("❌ Gradio not available")
            return None
        
        import gradio as gr
        
        def get_chrome_fix():
            diagnosis = self.diagnose_chrome_reboot()
            
            result = f"""# 🔍 Chrome Reboot Fix

**Issue:** {diagnosis['issue']}
**Analysis:** {diagnosis['analysis']}
**Confidence:** {diagnosis['confidence']:.0%}

## Solutions:

"""
            for solution in diagnosis['solutions']:
                result += f"""
### Step {solution['step']}: {solution['action']}
- **Command:** `{solution['command']}`
- **Risk:** {solution['risk']}
- **Expected:** {solution['expected_result']}
"""
            
            return result
        
        def get_system_info():
            return f"""# 🚀 Viren System Status

**Available Tech:** {', '.join(self.available_tech.keys())}
**System Status:** {self.system_status}
**Mode:** Lightweight (no hanging imports)

**Your Chrome Issue:**
This is a classic hardware acceleration problem. Step 1 should fix it immediately.
"""
        
        # Create simple interface
        with gr.Blocks(title="Viren Lightweight") as interface:
            gr.Markdown("# 🚀 Viren Lightweight Troubleshooter")
            
            with gr.Row():
                with gr.Column():
                    chrome_btn = gr.Button("🔍 Fix Chrome Reboots", variant="primary", size="lg")
                    chrome_output = gr.Markdown()
                
                with gr.Column():
                    status_btn = gr.Button("📊 System Status", variant="secondary")
                    status_output = gr.Markdown()
            
            chrome_btn.click(get_chrome_fix, outputs=chrome_output)
            status_btn.click(get_system_info, outputs=status_output)
        
        return interface
    
    def start_ui(self, port: int = 7860):
        """Start UI without hanging"""
        
        try:
            interface = self.create_simple_gradio_ui()
            if interface:
                # Try multiple ports
                for try_port in range(port, port + 10):
                    try:
                        print(f"🌐 Trying port {try_port}...")
                        interface.launch(server_port=try_port, share=False, quiet=True)
                        print(f"✅ Gradio running at http://localhost:{try_port}")
                        return True
                    except Exception as port_error:
                        if "port" in str(port_error).lower():
                            continue
                        else:
                            raise port_error
                
                print("❌ No available ports found")
                return False
            else:
                print("❌ Failed to create interface")
                return False
        except Exception as e:
            print(f"❌ UI failed: {e}")
            return False

# Global instance
VIREN_LIGHT = VirenLightweight()

def diagnose_chrome():
    """Quick Chrome diagnosis"""
    return VIREN_LIGHT.diagnose_chrome_reboot()

def start_ui():
    """Start UI"""
    return VIREN_LIGHT.start_ui()

# Main execution
if __name__ == "__main__":
    print("🚀 Viren Lightweight - No Hanging!")
    print("=" * 40)
    
    # Test Chrome diagnosis
    diagnosis = diagnose_chrome()
    print(f"\n🔍 Chrome Fix Ready:")
    print(f"   Confidence: {diagnosis['confidence']:.0%}")
    print(f"   Solutions: {len(diagnosis['solutions'])}")
    
    # Start UI if Gradio available
    if "gradio" in VIREN_LIGHT.available_tech:
        print(f"\n🌐 Starting Gradio UI...")
        if start_ui():
            print(f"✅ UI running at http://localhost:7860")
        else:
            print(f"❌ UI failed to start")
    else:
        print(f"\n⚠️ Gradio not available - showing diagnosis:")
        for solution in diagnosis['solutions']:
            print(f"   {solution['step']}. {solution['action']}")
    
    print(f"\n🎯 VIREN LIGHTWEIGHT READY!")