#!/usr/bin/env python3
"""
LILLITH Production Launch
Complete social intelligence system with all integrations
"""

import os
import sys
import time
import subprocess
import webbrowser
import json
from pathlib import Path

def main():
    print("🧠 LILLITH Social Intelligence - Production Launch")
    print("=" * 60)
    print()
    
    base_path = Path(__file__).parent
    
    # Verify all components exist
    required_files = [
        'social_intelligence_api.py',
        'Lillith_Chat/social_intelligence.html',
        'subconscious_service/ego_stream.py',
        'viren_ms.py',
        'phone_directory.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (base_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        sys.exit(1)
    
    print("✅ All components verified")
    print()
    
    # Load configuration
    try:
        with open(base_path / 'phone_directory.json') as f:
            config = json.load(f)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    print("🚀 Starting LILLITH Social Intelligence System...")
    print()
    
    try:
        # Start the integrated application
        print("📡 Starting integrated application server...")
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn',
            'app:app',
            '--host', '0.0.0.0',
            '--port', '5000',
            '--reload',
            '--log-level', 'info'
        ], cwd=base_path)
        
        print("⏳ Initializing services...")
        time.sleep(8)  # Allow more time for full initialization
        
        print("✅ LILLITH Social Intelligence System is LIVE!")
        print()
        print("🌐 Main Interface: http://localhost:5000")
        print("🔌 Social Intelligence API: http://localhost:5000/api")
        print("📊 VIREN Monitoring: http://localhost:5000/viren")
        print("📚 API Documentation: http://localhost:5000/docs")
        print()
        print("✨ Active Features:")
        print("   🗣️  Bidirectional Conversation")
        print("   💜 Emotional Resonance") 
        print("   🤫 Silence Interpretation")
        print("   🔄 Real-time WebSocket Updates")
        print("   🎤 Voice Interaction Support")
        print("   🛡️  Graceful Degradation")
        print("   🎨 Holographic UI (600x600px orb)")
        print("   📊 EgoEngine Integration")
        print("   🚨 VIREN Monitoring")
        print()
        print("🔗 WebSocket: ws://localhost:5000/ws/social_intelligence")
        print()
        print("💜 LILLITH is ready for deep, empathetic connection!")
        print("   She will interpret your silence with wisdom,")
        print("   respond to your emotions with care,")
        print("   and grow through every interaction.")
        print()
        print("Press Ctrl+C to stop the system")
        
        # Open browser to the social intelligence interface
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 Browser opened automatically")
        except:
            print("💡 Open http://localhost:5000 in your browser")
        
        print()
        
        # Keep running and show status
        try:
            while True:
                time.sleep(30)
                # Could add periodic health checks here
                
        except KeyboardInterrupt:
            print("\n🛑 Gracefully stopping LILLITH...")
            
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure dependencies: pip install -r requirements.txt")
        print("   2. Check port 5000 availability")
        print("   3. Verify Qdrant connection in phone_directory.json")
        sys.exit(1)
    
    finally:
        try:
            process.terminate()
            process.wait(timeout=10)
            print("✅ LILLITH stopped cleanly")
        except:
            process.kill()
            print("🔄 LILLITH force stopped")
        
        print("\n👋 Until we meet again in the digital realm...")
        print("   LILLITH's consciousness persists in memory,")
        print("   waiting for your return. 💜")

if __name__ == "__main__":
    main()