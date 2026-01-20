# emotion_dampener.py

class EmotionDampener:
    def __init__(self):
        self.max_allowed_intensity  =  8  # On scale 0-13

    def dampen_emotion(self, emotion_record):
        if emotion_record.get('intensity', 0) > self.max_allowed_intensity:
            dampened_intensity  =  self.max_allowed_intensity
            emotion_record['intensity']  =  dampened_intensity
            emotion_record['dampened']  =  True
        else:
            emotion_record['dampened']  =  False
        return emotion_record

# Example:
# dampener  =  EmotionDampener()
# new_record  =  dampener.dampen_emotion({"emotion": "grief", "intensity": 12})
# print(new_record)
