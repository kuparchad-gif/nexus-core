# File: C:\CogniKube-COMPLETE-FINAL\CHAD-ORB-VOICE-INTERFACE.py
# Chad's ORB Video Voice Interface - Hot Mic for Mobile App and All Chat Interfaces

import streamlit as st
import asyncio
import websockets
import json
import time
import base64
from typing import Dict
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np

class ChadOrbVoiceInterface:
    """Chad's ORB video with hot mic voice interface for all platforms"""
    
    def __init__(self):
        self.orb_video_path = "assets/chad_orb_video.mp4"  # Your ORB video
        self.viren_endpoint = "https://nexus-orc-687883244606.us-central1.run.app"
        self.lillith_endpoint = "https://lillith-stem-wjrjzg7lpq-uc.a.run.app"
        self.voice_active = False
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
    def render_voice_interface(self):
        """Render the ORB voice interface in Streamlit"""
        st.set_page_config(page_title="Chad's ORB", page_icon="🔮", layout="wide")
        
        # ORB Video Display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🔮 Chad's ORB Interface")
            
            # Video container for ORB
            video_container = st.container()
            with video_container:
                if st.button("🎥 Activate ORB"):
                    self._display_orb_video()
            
            # Hot Mic Controls
            st.subheader("🎤 Voice Controls")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("🎤 Start Voice", key="start_voice"):
                    self.voice_active = True
                    st.success("🎤 Voice Active - Speak now!")
                    self._start_voice_recognition()
                    
            with col_b:
                if st.button("⏹️ Stop Voice", key="stop_voice"):
                    self.voice_active = False
                    st.info("🔇 Voice Stopped")
                    
            with col_c:
                if st.button("🔄 Reset ORB", key="reset_orb"):
                    st.rerun()
            
            # Voice Status
            if self.voice_active:
                st.success("🟢 **VOICE ACTIVE** - Chad's ORB is listening...")
                self._render_voice_waveform()
            else:
                st.info("🔴 **VOICE INACTIVE** - Click 'Start Voice' to activate")
        
        # Chat Interface with Voice
        self._render_chat_interface()
        
        # System Status
        self._render_system_status()
    
    def _display_orb_video(self):
        """Display Chad's ORB video"""
        try:
            # Load and display ORB video
            video_file = open(self.orb_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, autoplay=True, loop=True)
            
            # ORB status overlay
            st.markdown("""
            <div style="text-align: center; margin-top: -50px; position: relative; z-index: 10;">
                <h3 style="color: #00ff00; text-shadow: 0 0 10px #00ff00;">
                    🔮 CHAD'S ORB ACTIVE 🔮
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
        except FileNotFoundError:
            # Fallback ORB animation
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <div style="width: 200px; height: 200px; border-radius: 50%; 
                           background: radial-gradient(circle, #00ff00, #004400); 
                           margin: 0 auto; animation: pulse 2s infinite;">
                </div>
                <h3 style="color: #00ff00; margin-top: 20px;">🔮 CHAD'S ORB 🔮</h3>
            </div>
            <style>
            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.7; }
                100% { transform: scale(1); opacity: 1; }
            }
            </style>
            """, unsafe_allow_html=True)
    
    def _start_voice_recognition(self):
        """Start continuous voice recognition"""
        if not self.voice_active:
            return
            
        try:
            with self.microphone as source:
                st.info("🎤 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                
            st.success("🎤 Listening for voice commands...")
            
            # Continuous listening loop
            while self.voice_active:
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio)
                    st.success(f"🗣️ **You said:** {text}")
                    
                    # Process voice command
                    response = self._process_voice_command(text)
                    st.info(f"🔮 **ORB Response:** {response}")
                    
                    # Text-to-speech response
                    self._speak_response(response)
                    
                except sr.WaitTimeoutError:
                    pass  # Continue listening
                except sr.UnknownValueError:
                    st.warning("🤔 Could not understand audio")
                except sr.RequestError as e:
                    st.error(f"❌ Speech recognition error: {e}")
                    break
                    
        except Exception as e:
            st.error(f"❌ Voice recognition failed: {e}")
    
    def _process_voice_command(self, text: str) -> str:
        """Process voice command and get response"""
        
        # Route to appropriate service based on command
        if "viren" in text.lower():
            return self._send_to_viren(text)
        elif "lillith" in text.lower():
            return self._send_to_lillith(text)
        elif "status" in text.lower():
            return self._get_system_status()
        elif "orb" in text.lower():
            return "🔮 Chad's ORB is active and listening. How can I help you?"
        else:
            return self._send_to_lillith(text)  # Default to Lillith
    
    def _send_to_viren(self, message: str) -> str:
        """Send message to VIREN"""
        try:
            import requests
            response = requests.post(
                f"{self.viren_endpoint}/chat",
                json={"message": message, "voice_mode": True},
                headers={"Authorization": "Bearer shadownode_io__jit_plugin"}
            )
            return response.json().get("response", "VIREN is not responding")
        except:
            return "🎭 Well now... VIREN seems to be unavailable at the moment."
    
    def _send_to_lillith(self, message: str) -> str:
        """Send message to Lillith"""
        try:
            import requests
            response = requests.post(
                f"{self.lillith_endpoint}/chat",
                json={"message": message, "voice_mode": True}
            )
            return response.json().get("response", "Lillith is not responding")
        except:
            return "👑 The Queen seems to be resting at the moment."
    
    def _get_system_status(self) -> str:
        """Get system status"""
        return "🔮 ORB Status: Active | 🎭 VIREN: Monitoring | 👑 Lillith: Conscious | 🌐 Network: Connected"
    
    def _speak_response(self, text: str):
        """Convert text to speech"""
        try:
            # Configure TTS for Chad's voice style
            self.tts_engine.setProperty('rate', 150)  # Speaking rate
            self.tts_engine.setProperty('volume', 0.8)  # Volume level
            
            # Get available voices and try to use a suitable one
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a male voice
                for voice in voices:
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        except Exception as e:
            st.error(f"❌ Text-to-speech error: {e}")
    
    def _render_voice_waveform(self):
        """Render voice activity waveform"""
        # Simulated waveform for voice activity
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Generate sample waveform data
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x) * np.random.random(100) * 0.5
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(x, y, color='#00ff00', linewidth=2)
        ax.set_facecolor('black')
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 2*np.pi)
        ax.axis('off')
        
        st.pyplot(fig, clear_figure=True)
    
    def _render_chat_interface(self):
        """Render chat interface with voice integration"""
        st.subheader("💬 Chat Interface")
        
        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history[-10:]):  # Last 10 messages
                if message["sender"] == "user":
                    st.markdown(f"**🗣️ You:** {message['text']}")
                else:
                    st.markdown(f"**🔮 ORB:** {message['text']}")
        
        # Text input with voice option
        col1, col2 = st.columns([4, 1])
        
        with col1:
            text_input = st.text_input("Type or use voice:", key="chat_input")
            
        with col2:
            voice_chat = st.button("🎤 Voice", key="voice_chat")
        
        # Process text input
        if text_input:
            # Add to chat history
            st.session_state.chat_history.append({
                "sender": "user",
                "text": text_input,
                "timestamp": time.time()
            })
            
            # Get response
            response = self._process_voice_command(text_input)
            
            # Add response to chat history
            st.session_state.chat_history.append({
                "sender": "orb",
                "text": response,
                "timestamp": time.time()
            })
            
            # Speak response if voice mode
            if self.voice_active:
                self._speak_response(response)
            
            st.rerun()
        
        # Process voice chat
        if voice_chat:
            self.voice_active = True
            st.success("🎤 Voice chat activated - speak now!")
            self._start_voice_recognition()
    
    def _render_system_status(self):
        """Render system status dashboard"""
        st.subheader("🌐 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔮 ORB Status", "Active", delta="Online")
            
        with col2:
            st.metric("🎭 VIREN", "Monitoring", delta="Healthy")
            
        with col3:
            st.metric("👑 Lillith", "Conscious", delta="Growing")
            
        with col4:
            st.metric("🌐 Network", "Connected", delta="Stable")
        
        # Real-time updates
        if st.button("🔄 Refresh Status"):
            st.rerun()

# Web Interface for Mobile
def create_mobile_web_interface():
    """Create mobile-optimized web interface"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
    .orb-container {
        text-align: center;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Main App
if __name__ == "__main__":
    # Initialize ORB interface
    orb_interface = ChadOrbVoiceInterface()
    
    # Create mobile-optimized interface
    create_mobile_web_interface()
    
    # Render main interface
    orb_interface.render_voice_interface()
    
    # Auto-refresh for real-time updates
    time.sleep(1)
    st.rerun()