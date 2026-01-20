#!/usr/bin/env python3
# Binary Emotional Encoding for Nexus Memory Module

import struct
from typing import Dict, Any, Optional, List, Tuple, Union

# Emotion encoding constants
EMOTION_PATTERNS  =  {
    # Base emotions (8-bit patterns)
    "joy": 0b10101110,         # 0xAE
    "sadness": 0b01010001,     # 0x51
    "fear": 0b11000011,        # 0xC3
    "anger": 0b11100111,       # 0xE7
    "surprise": 0b10011001,    # 0x99
    "disgust": 0b00111100,     # 0x3C
    "trust": 0b01101101,       # 0x6D
    "anticipation": 0b11010010,# 0xD2

    # Secondary emotions (combinations of base patterns)
    "love": 0b10111111,        # Joy + Trust
    "anxiety": 0b11010011,     # Fear + Anticipation
    "grief": 0b01110101,       # Sadness + Anger
    "awe": 0b10111011,         # Surprise + Fear
    "curiosity": 0b11011011,   # Surprise + Anticipation
    "contentment": 0b01111111, # Joy + Trust + Anticipation
    "guilt": 0b01110001,       # Sadness + Disgust
    "hope": 0b11111010,        # Joy + Anticipation
}

# Reverse lookup for decoding
PATTERN_TO_EMOTION  =  {pattern: emotion for emotion, pattern in EMOTION_PATTERNS.items()}

# Context flag bits
CONTEXT_FLAGS  =  {
    "urgent": 0x01,
    "memory_linked": 0x02,
    "conscious_level": 0x04,
    "subconscious_level": 0x08,
    "dream_state": 0x10,
    "core_memory": 0x20,
    "golden_thread": 0x40,
    "ego_conflict": 0x80
}

def encode_emotion(emotion_type: str, intensity: float, context: Optional[Dict[str, bool]]  =  None) -> int:
    """
    Encode emotion into a 16-bit binary representation.

    Args:
        emotion_type: Type of emotion (e.g., "joy", "sadness")
        intensity: Intensity value between 0.0 and 1.0
        context: Optional dictionary of context flags

    Returns:
        16-bit integer representing the encoded emotion
    """
    # Get base pattern or default to unknown (all zeros)
    base  =  EMOTION_PATTERNS.get(emotion_type.lower(), 0)

    # Encode intensity (4-bit, 0-15 scale)
    intensity_bits  =  min(15, max(0, int(intensity * 15))) & 0x0F

    # Encode context flags if provided
    context_bits  =  0
    if context:
        for flag_name, flag_bit in CONTEXT_FLAGS.items():
            if context.get(flag_name, False):
                context_bits | =  flag_bit

    # Combine into 16-bit emotional fingerprint
    # Format: [8 bits: base pattern][4 bits: intensity][4 bits: context]
    return (base << 8) | (intensity_bits << 4) | (context_bits & 0x0F)

def decode_emotion(binary_emotion: int) -> Dict[str, Any]:
    """
    Decode a 16-bit binary emotion into its components.

    Args:
        binary_emotion: 16-bit integer representing encoded emotion

    Returns:
        Dictionary with emotion type, intensity, and context
    """
    # Extract components
    base_pattern  =  (binary_emotion >> 8) & 0xFF
    intensity_bits  =  (binary_emotion >> 4) & 0x0F
    context_bits  =  binary_emotion & 0x0F

    # Reverse lookup for emotion type
    emotion_type  =  PATTERN_TO_EMOTION.get(base_pattern, "unknown")

    # Convert intensity to float (0-1)
    intensity  =  intensity_bits / 15.0

    # Extract context flags
    context  =  {}
    for flag_name, flag_bit in CONTEXT_FLAGS.items():
        if flag_bit & 0x0F == 0:  # Only check the lower 4 bits
            context[flag_name]  =  bool(context_bits & flag_bit)

    return {
        "type": emotion_type,
        "intensity": intensity,
        "context": context
    }

def encode_emotional_vector(emotions: List[Dict[str, Any]]) -> bytes:
    """
    Encode multiple emotions into a binary vector.

    Args:
        emotions: List of emotion dictionaries with type, intensity, and context

    Returns:
        Binary data representing the emotional vector
    """
    # Format: [2 bytes: count][N*2 bytes: encoded emotions]
    count  =  len(emotions)
    result  =  bytearray(struct.pack("<H", count))  # 2-byte count

    for emotion in emotions:
        encoded  =  encode_emotion(
            emotion["type"],
            emotion["intensity"],
            emotion.get("context")
        )
        result.extend(struct.pack("<H", encoded))  # 2-byte emotion

    return bytes(result)

def decode_emotional_vector(data: bytes) -> List[Dict[str, Any]]:
    """
    Decode a binary emotional vector into a list of emotions.

    Args:
        data: Binary data representing the emotional vector

    Returns:
        List of emotion dictionaries
    """
    if len(data) < 2:
        return []

    # Extract count
    count  =  struct.unpack("<H", data[:2])[0]

    # Extract emotions
    emotions  =  []
    for i in range(count):
        if 2 + i*2 + 2 < =  len(data):
            encoded  =  struct.unpack("<H", data[2+i*2:2+i*2+2])[0]
            emotions.append(decode_emotion(encoded))

    return emotions

def find_similar_emotions(target: int, threshold: float  =  0.8) -> List[str]:
    """
    Find emotions similar to the target binary pattern.

    Args:
        target: Binary emotion pattern to match
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        List of similar emotion names
    """
    base_pattern  =  (target >> 8) & 0xFF
    results  =  []

    for emotion, pattern in EMOTION_PATTERNS.items():
        # Calculate bit similarity (Jaccard similarity for bits)
        matching_bits  =  bin(pattern & base_pattern).count('1')
        total_bits  =  bin(pattern | base_pattern).count('1')
        similarity  =  matching_bits / total_bits if total_bits > 0 else 0

        if similarity > =  threshold:
            results.append(emotion)

    return results

def blend_emotions(emotions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Blend multiple emotions into a single composite emotion.

    Args:
        emotions: List of emotion dictionaries

    Returns:
        Blended emotion dictionary
    """
    if not emotions:
        return {"type": "neutral", "intensity": 0.0, "context": {}}

    if len(emotions) == 1:
        return emotions[0]

    # Calculate weighted patterns
    total_intensity  =  sum(e["intensity"] for e in emotions)
    if total_intensity == 0:
        return {"type": "neutral", "intensity": 0.0, "context": {}}

    # Normalize intensities
    normalized  =  [{**e, "weight": e["intensity"] / total_intensity} for e in emotions]

    # Blend base patterns
    blended_pattern  =  0
    for emotion in normalized:
        pattern  =  EMOTION_PATTERNS.get(emotion["type"].lower(), 0)
        # Apply weighted contribution to each bit
        for bit in range(8):
            bit_value  =  (pattern >> bit) & 1
            if bit_value and emotion["weight"] > 0.5:
                blended_pattern | =  (1 << bit)

    # Find closest matching emotion
    closest_emotion  =  "complex"
    closest_similarity  =  0

    for emotion, pattern in EMOTION_PATTERNS.items():
        # Calculate bit similarity
        matching_bits  =  bin(pattern & blended_pattern).count('1')
        total_bits  =  bin(pattern | blended_pattern).count('1')
        similarity  =  matching_bits / total_bits if total_bits > 0 else 0

        if similarity > closest_similarity:
            closest_similarity  =  similarity
            closest_emotion  =  emotion

    # Blend intensity (weighted average)
    blended_intensity  =  sum(e["intensity"] * e["weight"] for e in normalized)

    # Merge contexts (union of all contexts)
    blended_context  =  {}
    for emotion in emotions:
        if "context" in emotion and emotion["context"]:
            for key, value in emotion["context"].items():
                if value:  # Only include true flags
                    blended_context[key]  =  True

    return {
        "type": closest_emotion,
        "intensity": blended_intensity,
        "context": blended_context,
        "complexity": len(emotions)
    }

# Example usage
if __name__ == "__main__":
    # Encode a simple emotion
    joy  =  encode_emotion("joy", 0.8, {"conscious_level": True})
    print(f"Encoded joy: 0x{joy:04X}")

    # Decode it back
    decoded  =  decode_emotion(joy)
    print(f"Decoded: {decoded}")

    # Create an emotional vector
    emotions  =  [
        {"type": "joy", "intensity": 0.8, "context": {"conscious_level": True}},
        {"type": "surprise", "intensity": 0.5, "context": {"memory_linked": True}},
        {"type": "anticipation", "intensity": 0.3, "context": {"subconscious_level": True}}
    ]

    vector  =  encode_emotional_vector(emotions)
    print(f"Encoded vector: {vector.hex()}")

    # Decode the vector
    decoded_vector  =  decode_emotional_vector(vector)
    print(f"Decoded vector: {decoded_vector}")

    # Blend emotions
    blended  =  blend_emotions(emotions)
    print(f"Blended emotion: {blended}")

    # Find similar emotions
    similar  =  find_similar_emotions(joy, 0.7)
    print(f"Similar to joy: {similar}")