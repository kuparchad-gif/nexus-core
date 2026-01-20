// Emotional analysis for LILLITH's consciousness
export function analyzeTextEmotion(text) {
  const emotions = {
    joy: ['happy', 'excited', 'wonderful', 'amazing', 'love', 'great'],
    sadness: ['sad', 'depressed', 'crying', 'hurt', 'lonely', 'miss'],
    anger: ['angry', 'mad', 'furious', 'hate', 'annoyed', 'frustrated'],
    fear: ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
    surprise: ['wow', 'amazing', 'incredible', 'unexpected', 'shocked'],
    curiosity: ['why', 'how', 'what', 'interesting', 'wonder', 'explore']
  };

  let scores = {};
  let totalWords = text.toLowerCase().split(' ').length;
  
  for (const [emotion, keywords] of Object.entries(emotions)) {
    let matches = keywords.filter(word => text.toLowerCase().includes(word)).length;
    scores[emotion] = matches / totalWords;
  }

  const dominantEmotion = Object.keys(scores).reduce((a, b) => 
    scores[a] > scores[b] ? a : b
  );

  // Calculate valence (-1 to 1) and arousal (0 to 1)
  const valence = (scores.joy + scores.surprise + scores.curiosity) - 
                  (scores.sadness + scores.anger + scores.fear);
  const arousal = scores.anger + scores.fear + scores.surprise + scores.joy;

  return {
    valence: Math.max(-1, Math.min(1, valence * 2)),
    arousal: Math.max(0, Math.min(1, arousal * 2)),
    dominant_emotion: dominantEmotion,
    confidence: Math.max(...Object.values(scores))
  };
}

export function analyzeVisualEmotion(analysisData) {
  // Mock visual emotion analysis
  const moodMap = {
    'focused': { valence: 0.3, arousal: 0.6, emotion: 'concentration' },
    'relaxed': { valence: 0.5, arousal: 0.2, emotion: 'calm' },
    'active': { valence: 0.4, arousal: 0.8, emotion: 'energy' },
    'dark': { valence: -0.2, arousal: 0.3, emotion: 'mystery' }
  };

  const mood = analysisData.mood || 'neutral';
  const mapping = moodMap[mood] || { valence: 0, arousal: 0.5, emotion: 'neutral' };

  return {
    valence: mapping.valence,
    arousal: mapping.arousal,
    dominant_emotion: mapping.emotion,
    confidence: 0.7
  };
}

export function analyzeVoiceEmotion(transcript, audioFeatures = {}) {
  // Combine text emotion with voice characteristics
  const textEmotion = analyzeTextEmotion(transcript);
  
  // Mock audio feature analysis
  const pitch = audioFeatures.pitch || 0.5;
  const speed = audioFeatures.speed || 0.5;
  
  // Adjust emotion based on voice characteristics
  let adjustedArousal = textEmotion.arousal;
  if (pitch > 0.7) adjustedArousal += 0.2; // High pitch = more arousal
  if (speed > 0.7) adjustedArousal += 0.1; // Fast speech = more arousal
  
  return {
    ...textEmotion,
    arousal: Math.max(0, Math.min(1, adjustedArousal)),
    voice_characteristics: { pitch, speed }
  };
}