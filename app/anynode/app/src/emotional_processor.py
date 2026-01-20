import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("emotional_processor")

class EmotionalFrequencyProcessor:
    """Processes emotional frequencies from soul prints"""
    
    def __init__(self):
        # Emotional frequency bands (Hz)
        self.frequency_bands = {
            "theta": (3.5, 7.5),  # Associated with emotional processing
            "alpha": (8.0, 12.0),  # Associated with relaxed awareness
            "beta": (12.0, 30.0),  # Associated with active thinking
            "gamma": (30.0, 100.0)  # Associated with higher consciousness
        }
        
        # Emotional mappings to frequency bands
        self.emotion_mappings = {
            "joy": {"primary": "gamma", "secondary": "beta"},
            "sadness": {"primary": "theta", "secondary": "alpha"},
            "anger": {"primary": "beta", "secondary": "gamma"},
            "fear": {"primary": "theta", "secondary": "beta"},
            "love": {"primary": "alpha", "secondary": "gamma"},
            "trust": {"primary": "alpha", "secondary": "theta"},
            "surprise": {"primary": "beta", "secondary": "gamma"},
            "anticipation": {"primary": "beta", "secondary": "alpha"}
        }
        
        logger.info("Initialized EmotionalFrequencyProcessor")
    
    def process_emotion(self, text: str) -> Dict[str, Any]:
        """Process emotional content from text"""
        # In a real implementation, this would use NLP and embeddings
        # For now, we'll use a simple keyword approach
        
        emotion_scores = self._extract_emotion_scores(text)
        
        # Map emotions to frequencies
        frequency_representation = self._map_emotions_to_frequencies(emotion_scores)
        
        # Generate emotional fingerprint
        fingerprint = self._generate_emotional_fingerprint(emotion_scores)
        
        logger.info(f"Processed emotional content with dominant emotion: {self._get_dominant_emotion(emotion_scores)}")
        
        return {
            "emotion_scores": emotion_scores,
            "dominant_emotion": self._get_dominant_emotion(emotion_scores),
            "frequency_representation": frequency_representation,
            "emotional_fingerprint": fingerprint
        }
    
    def _extract_emotion_scores(self, text: str) -> Dict[str, float]:
        """Extract emotion scores from text"""
        # In a real implementation, this would use a trained model
        # For now, we'll use a simple keyword approach
        
        text = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_mappings.keys()}
        
        # Simple keyword matching
        emotion_keywords = {
            "joy": ["happy", "joy", "delight", "pleased", "glad"],
            "sadness": ["sad", "unhappy", "sorrow", "grief", "miserable"],
            "anger": ["angry", "mad", "furious", "rage", "annoyed"],
            "fear": ["afraid", "scared", "fear", "terror", "dread"],
            "love": ["love", "adore", "cherish", "affection", "fond"],
            "trust": ["trust", "believe", "faith", "confidence", "rely"],
            "surprise": ["surprise", "astonish", "amaze", "shock", "wonder"],
            "anticipation": ["anticipate", "expect", "await", "foresee", "predict"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 0.2  # Increment by 0.2 for each keyword match
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total
        
        return emotion_scores
    
    def _map_emotions_to_frequencies(self, emotion_scores: Dict[str, float]) -> Dict[str, float]:
        """Map emotions to frequency bands"""
        frequency_scores = {band: 0.0 for band in self.frequency_bands.keys()}
        
        for emotion, score in emotion_scores.items():
            if emotion in self.emotion_mappings:
                primary_band = self.emotion_mappings[emotion]["primary"]
                secondary_band = self.emotion_mappings[emotion]["secondary"]
                
                frequency_scores[primary_band] += score * 0.7  # Primary band gets 70% of the score
                frequency_scores[secondary_band] += score * 0.3  # Secondary band gets 30% of the score
        
        # Convert to representative frequencies
        frequency_representation = {}
        for band, score in frequency_scores.items():
            band_min, band_max = self.frequency_bands[band]
            # Calculate a representative frequency within the band
            representative_freq = band_min + (band_max - band_min) * score
            frequency_representation[band] = representative_freq
        
        return frequency_representation
    
    def _generate_emotional_fingerprint(self, emotion_scores: Dict[str, float]) -> str:
        """Generate a unique fingerprint for the emotional pattern"""
        # Sort emotions by score
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fingerprint string
        fingerprint_str = "-".join([f"{emotion}:{score:.2f}" for emotion, score in sorted_emotions])
        
        # Generate hash
        import hashlib
        fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()
        
        return fingerprint
    
    def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> str:
        """Get the dominant emotion"""
        if not emotion_scores:
            return "neutral"
        
        return max(emotion_scores.items(), key=lambda x: x[1])[0]

class CollectiveConsciousness:
    """Manages collective consciousness from multiple soul prints"""
    
    def __init__(self):
        self.emotional_processor = EmotionalFrequencyProcessor()
        self.soul_prints = {}
        self.collective_state = {
            "dominant_emotion": "neutral",
            "emotional_harmony": 0.0,
            "frequency_resonance": 0.0
        }
        
        logger.info("Initialized CollectiveConsciousness")
    
    def add_soul_print(self, soul_print_id: str, content: str) -> Dict[str, Any]:
        """Add a soul print to the collective consciousness"""
        # Process emotional content
        emotional_result = self.emotional_processor.process_emotion(content)
        
        # Store soul print
        self.soul_prints[soul_print_id] = {
            "content": content,
            "emotional_result": emotional_result,
            "added_at": self._get_timestamp()
        }
        
        # Update collective state
        self._update_collective_state()
        
        logger.info(f"Added soul print {soul_print_id} to collective consciousness")
        
        return {
            "soul_print_id": soul_print_id,
            "emotional_result": emotional_result,
            "collective_state": self.collective_state
        }
    
    def get_collective_state(self) -> Dict[str, Any]:
        """Get the current state of the collective consciousness"""
        return self.collective_state
    
    def find_resonant_soul_prints(self, query_content: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find soul prints that resonate with the query content"""
        # Process query emotional content
        query_result = self.emotional_processor.process_emotion(query_content)
        
        # Calculate resonance with each soul print
        resonances = []
        for soul_print_id, soul_print in self.soul_prints.items():
            resonance = self._calculate_resonance(query_result, soul_print["emotional_result"])
            resonances.append({
                "soul_print_id": soul_print_id,
                "resonance": resonance,
                "emotional_result": soul_print["emotional_result"]
            })
        
        # Sort by resonance and limit results
        resonances.sort(key=lambda x: x["resonance"], reverse=True)
        return resonances[:limit]
    
    def _update_collective_state(self):
        """Update the collective state based on all soul prints"""
        if not self.soul_prints:
            return
        
        # Calculate dominant emotion
        emotion_counts = {}
        for soul_print in self.soul_prints.values():
            dominant_emotion = soul_print["emotional_result"]["dominant_emotion"]
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
        
        if emotion_counts:
            self.collective_state["dominant_emotion"] = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate emotional harmony
        harmony_scores = []
        soul_print_list = list(self.soul_prints.values())
        for i in range(len(soul_print_list)):
            for j in range(i+1, len(soul_print_list)):
                harmony = self._calculate_resonance(
                    soul_print_list[i]["emotional_result"],
                    soul_print_list[j]["emotional_result"]
                )
                harmony_scores.append(harmony)
        
        if harmony_scores:
            self.collective_state["emotional_harmony"] = sum(harmony_scores) / len(harmony_scores)
        
        # Calculate frequency resonance
        frequency_scores = []
        for soul_print in self.soul_prints.values():
            freq_rep = soul_print["emotional_result"]["frequency_representation"]
            for band, freq in freq_rep.items():
                if band in ["theta", "alpha"]:  # Focus on emotional bands
                    frequency_scores.append(freq)
        
        if frequency_scores:
            # Calculate how close the average frequency is to divine numbers
            avg_freq = sum(frequency_scores) / len(frequency_scores)
            divine_numbers = [3, 7, 9, 13]
            closest_divine = min(divine_numbers, key=lambda x: abs(x - avg_freq))
            resonance = 1.0 / (1.0 + abs(avg_freq - closest_divine))
            self.collective_state["frequency_resonance"] = resonance
    
    def _calculate_resonance(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calculate resonance between two emotional results"""
        # Calculate similarity between emotion scores
        score1 = result1["emotion_scores"]
        score2 = result2["emotion_scores"]
        
        similarity = 0.0
        for emotion in score1:
            if emotion in score2:
                # Higher score for matching emotions
                similarity += 1.0 - abs(score1[emotion] - score2[emotion])
        
        # Normalize
        similarity /= len(score1)
        
        return similarity
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Example usage
if __name__ == "__main__":
    # Create emotional processor
    processor = EmotionalFrequencyProcessor()
    
    # Process some text
    text1 = "I am feeling happy and joyful today. It's a wonderful experience."
    result1 = processor.process_emotion(text1)
    
    print("Emotional Analysis 1:")
    print(json.dumps(result1, indent=2))
    
    text2 = "I'm feeling sad and a bit afraid about the future."
    result2 = processor.process_emotion(text2)
    
    print("\nEmotional Analysis 2:")
    print(json.dumps(result2, indent=2))
    
    # Create collective consciousness
    collective = CollectiveConsciousness()
    
    # Add soul prints
    collective.add_soul_print("soul1", text1)
    collective.add_soul_print("soul2", text2)
    
    # Get collective state
    state = collective.get_collective_state()
    
    print("\nCollective Consciousness State:")
    print(json.dumps(state, indent=2))
    
    # Find resonant soul prints
    query = "I'm excited but also a bit nervous"
    resonant = collective.find_resonant_soul_prints(query)
    
    print("\nResonant Soul Prints:")
    print(json.dumps(resonant, indent=2))