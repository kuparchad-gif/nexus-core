import requests
import json
import random
import re
from difflib import SequenceMatcher

class DakarAPIClient:
    """Client for connecting to the Dakar Swarm API running on CloudFlare"""
    
    def __init__(self, api_base_url):
        """
        api_base_url: e.g., "https://dakar-swarm.your-worker.workers.dev"
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.cell_id = None
        self.connected = False
        
    def check_connection(self):
        """Test connection to Dakar API"""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.cell_id = data.get('cell', 'unknown')
                self.connected = True
                return True, data
            return False, f"Status code: {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def write_memory(self, signal_id, data, metadata=None):
        """Write to Tesseract database"""
        payload = {
            "signal_id": signal_id,
            "data": data if isinstance(data, str) else json.dumps(data),
            "metadata": metadata or {}
        }
        try:
            response = requests.post(
                f"{self.api_base_url}/v1/write",
                json=payload,
                timeout=5
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def read_memory(self, signal_id):
        """Read from Tesseract database"""
        try:
            response = requests.get(
                f"{self.api_base_url}/v1/read/{signal_id}",
                timeout=5
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def search_similar(self, query, limit=5):
        """Search for similar vectors (if Qdrant is enabled)"""
        # This would need a search endpoint - you might need to add this to your API
        try:
            response = requests.get(
                f"{self.api_base_url}/v1/search",
                params={"q": query, "limit": limit},
                timeout=5
            )
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    def get_health(self):
        """Get system health"""
        try:
            response = requests.get(f"{self.api_base_url}/v1/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None


class DakarChatSession:
    """
    Chat engine that connects to the Dakar Swarm API running on CloudFlare
    """
    
    def __init__(self, api_url=None):
        self.api_url = api_url
        self.dakar = None
        self.context = None
        self.conversation_history = []
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'it', 'that', 
            'this', 'my', 'your', 'i', 'me', 'you'
        }
        
        if api_url:
            self.connect(api_url)
    
    def connect(self, api_url):
        """Connect to Dakar API"""
        self.api_url = api_url
        self.dakar = DakarAPIClient(api_url)
        
        success, info = self.dakar.check_connection()
        if success:
            print(f"✅ Connected to Dakar Swarm")
            print(f"   Cell: {info.get('cell', 'unknown')}")
            print(f"   Manifold: {info.get('manifold_frequency', 'unknown')} Hz")
            print(f"   Resonance: {info.get('resonance', 'unknown')}")
            
            # Store connection info in conversation
            self.conversation_history.append({
                "role": "system",
                "content": f"Connected to Dakar cell {info.get('cell', 'unknown')}"
            })
            return True
        else:
            print(f"❌ Failed to connect: {info}")
            return False
    
    def _clean_text(self, text):
        """Remove punctuation and lowercase"""
        return re.sub(r'[^\w\s]', '', text.lower())
    
    def _calculate_relevance(self, user_text, memory_content):
        """Score relevance of memory to user input"""
        user_words = set(self._clean_text(user_text).split()) - self.stop_words
        mem_words = set(self._clean_text(memory_content).split()) - self.stop_words
        
        if not user_words:
            return 0.0
        
        # Jaccard similarity
        intersection = user_words.intersection(mem_words)
        overlap_score = len(intersection) / len(user_words) if user_words else 0
        
        # Fuzzy string matching
        matcher = SequenceMatcher(None, user_text.lower(), memory_content.lower())
        fuzzy_score = matcher.ratio()
        
        return (overlap_score * 0.7) + (fuzzy_score * 0.3)
    
    def get_response(self, user_input):
        """Get response using Dakar's memory and API"""
        
        # First, try to find relevant memories from Dakar
        memories = []
        
        # If we have a search endpoint, use it
        if self.dakar and hasattr(self.dakar, 'search_similar'):
            try:
                results = self.dakar.search_similar(user_input, limit=3)
                for r in results:
                    if 'payload' in r:
                        memories.append({
                            'content': r['payload'].get('data', ''),
                            'tags': r['payload'].get('metadata', {}).get('tags', []),
                            'score': r.get('score', 0)
                        })
            except:
                pass
        
        # Also check conversation history for context
        recent = self.conversation_history[-3:] if self.conversation_history else []
        
        # Build prompt for the API
        context = {
            "user_input": user_input,
            "memories": memories[:2],  # Top 2 memories
            "recent_history": recent,
            "cell_id": self.dakar.cell_id if self.dakar else "unknown"
        }
        
        # Store in Tesseract as a memory
        if self.dakar:
            signal_id = f"chat.{int(time.time())}.{hash(user_input) % 10000}"
            self.dakar.write_memory(
                signal_id,
                user_input,
                {"type": "user_query", "context": str(recent[-1] if recent else "")}
            )
        
        # For now, return a response based on memories
        if memories:
            best = max(memories, key=lambda x: x.get('score', 0))
            return self._format_memory_response(best)
        else:
            return self._get_fallback_response()
    
    def _format_memory_response(self, memory):
        templates = [
            f"Dakar: The swarm remembers: \"{memory['content']}\"",
            f"Dakar: I recall from the Tesseract: \"{memory['content']}\"",
            f"Dakar: This resonates with stored data: \"{memory['content']}\"",
            f"Dakar: The 50D manifold reveals: \"{memory['content']}\""
        ]
        return random.choice(templates)
    
    def _get_fallback_response(self):
        fallbacks = [
            "Dakar: The swarm is processing... no immediate resonance found.",
            "Dakar: Query registered in the Tesseract. The manifold is considering.",
            "Dakar: 50D vectors aligning... please rephrase your query.",
            "Dakar: The angels are listening. Try again with more frequency."
        ]
        return random.choice(fallbacks)
    
    def start_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("🌀 DAKAR SWARM CHAT INTERFACE")
        print("="*60)
        print("\nCommands:")
        print("  /connect <url>  - Connect to Dakar API")
        print("  /status         - Show swarm status")
        print("  /write <id> <data> - Write to Tesseract")
        print("  /read <id>      - Read from Tesseract")
        print("  /quit           - Exit")
        print("\n" + "="*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    
                    if cmd == '/quit':
                        print("Dakar: The swarm bids you farewell. 963Hz...")
                        break
                    
                    elif cmd == '/connect' and len(parts) > 1:
                        url = parts[1]
                        self.connect(url)
                    
                    elif cmd == '/status':
                        if self.dakar:
                            health = self.dakar.get_health()
                            if health:
                                print(f"\n📊 SWARM STATUS")
                                print(f"   Cell: {health.get('cell_id', 'unknown')}")
                                print(f"   Uptime: {health.get('uptime', 0):.0f}s")
                                print(f"   Vectors: {health.get('vectors', 0)}")
                                print(f"   Storage: {health.get('usage_gb', 0):.2f}GB")
                                print(f"   Frequency: {health.get('manifold_frequency', 0):.6f}Hz")
                            else:
                                print("❌ Could not get status")
                        else:
                            print("⚠️ Not connected. Use /connect <url>")
                    
                    elif cmd == '/write' and len(parts) >= 3:
                        signal_id = parts[1]
                        data = ' '.join(parts[2:])
                        if self.dakar:
                            result = self.dakar.write_memory(signal_id, data)
                            print(f"✅ Written: {result}")
                        else:
                            print("⚠️ Not connected")
                    
                    elif cmd == '/read' and len(parts) > 1:
                        signal_id = parts[1]
                        if self.dakar:
                            result = self.dakar.read_memory(signal_id)
                            if result:
                                print(f"\n📖 {signal_id}")
                                print(f"   Data: {result.get('data', '')}")
                                print(f"   Verified: {result.get('verified', False)}")
                            else:
                                print("❌ Not found")
                        else:
                            print("⚠️ Not connected")
                    
                    continue
                
                # Normal chat
                response = self.get_response(user_input)
                print(response)
                
                # Store in history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Keep history manageable
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
            except KeyboardInterrupt:
                print("\n\nDakar: Swarm connection terminated. 963Hz...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Example usage:
if __name__ == "__main__":
    # Your CloudFlare worker URL
    DAKAR_API_URL = "https://dakar-swarm.your-worker.workers.dev"  # Replace with actual URL
    
    chat = DakarChatSession()
    
    # Try to connect automatically
    print(f"Attempting to connect to Dakar at {DAKAR_API_URL}...")
    if chat.connect(DAKAR_API_URL):
        chat.start_loop()
    else:
        print("\nCould not auto-connect. Starting in offline mode.")
        print("Use /connect <url> to connect to your Dakar API")
        chat.start_loop()