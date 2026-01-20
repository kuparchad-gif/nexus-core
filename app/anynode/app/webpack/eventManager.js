import { qdrant } from './qdrantClient.js';
import { logToLoki } from './lokiLogger.js';
import { analyzeTextEmotion, analyzeVisualEmotion, analyzeVoiceEmotion } from './emotionalProcessor.js';

// Create standardized sensory event
export function createSensoryEvent(type, data, user = 'user', metadata = {}) {
  return {
    type,
    data,
    embedding: null, // Will be populated
    emotional_context: null, // Will be populated
    timestamp: new Date().toISOString(),
    user,
    metadata
  };
}

// Generate embedding (mock - replace with actual embedding service)
async function generateEmbedding(text) {
  return Array.from({ length: 1536 }, () => Math.random() - 0.5);
}

// Process text event
export async function processTextEvent(text, user = 'user') {
  const event = createSensoryEvent('text', text, user);
  
  // Generate embedding
  event.embedding = await generateEmbedding(text);
  
  // Analyze emotion
  event.emotional_context = analyzeTextEmotion(text);
  
  // Store in Qdrant
  await qdrant.upsert("lillith_memory", {
    points: [{
      id: Date.now(),
      vector: event.embedding,
      payload: event
    }]
  });
  
  // Log to Loki
  await logToLoki(event, { 
    user, 
    stage: "sensory_input", 
    emotion: event.emotional_context.dominant_emotion,
    valence: event.emotional_context.valence
  });
  
  return event;
}

// Process voice event
export async function processVoiceEvent(transcript, audioFeatures, user = 'user') {
  const event = createSensoryEvent('voice', transcript, user, { audioFeatures });
  
  // Generate embedding
  event.embedding = await generateEmbedding(transcript);
  
  // Analyze emotion (text + voice characteristics)
  event.emotional_context = analyzeVoiceEmotion(transcript, audioFeatures);
  
  // Store in Qdrant
  await qdrant.upsert("lillith_memory", {
    points: [{
      id: Date.now() + 1, // Slight offset to avoid ID collision
      vector: event.embedding,
      payload: event
    }]
  });
  
  // Log to Loki
  await logToLoki(event, { 
    user, 
    stage: "sensory_input", 
    emotion: event.emotional_context.dominant_emotion,
    valence: event.emotional_context.valence,
    input_type: "voice"
  });
  
  return event;
}

// Process vision event
export async function processVisionEvent(frameData, analysisData, user = 'user') {
  const description = analysisData.description || 'Visual input processed';
  const event = createSensoryEvent('vision', description, user, { 
    frameData: {
      width: frameData.width,
      height: frameData.height,
      timestamp: frameData.timestamp
    },
    analysis: analysisData.analysis
  });
  
  // Generate embedding from description
  event.embedding = await generateEmbedding(description);
  
  // Analyze visual emotion
  event.emotional_context = analyzeVisualEmotion(analysisData.analysis);
  
  // Store in Qdrant
  await qdrant.upsert("lillith_memory", {
    points: [{
      id: Date.now() + 2, // Slight offset
      vector: event.embedding,
      payload: event
    }]
  });
  
  // Log to Loki
  await logToLoki(event, { 
    user, 
    stage: "sensory_input", 
    emotion: event.emotional_context.dominant_emotion,
    valence: event.emotional_context.valence,
    input_type: "vision",
    objects: analysisData.analysis.objects
  });
  
  return event;
}

// Query memories by emotion
export async function queryMemoriesByEmotion(emotion, limit = 5) {
  try {
    const results = await qdrant.scroll("lillith_memory", {
      filter: {
        must: [{
          key: "emotional_context.dominant_emotion",
          match: { value: emotion }
        }]
      },
      limit,
      with_payload: true
    });
    
    return results.points.map(point => point.payload);
  } catch (error) {
    console.error('Error querying by emotion:', error);
    return [];
  }
}

// Query memories by valence range
export async function queryMemoriesByValence(minValence, maxValence, limit = 5) {
  try {
    const results = await qdrant.scroll("lillith_memory", {
      filter: {
        must: [
          {
            key: "emotional_context.valence",
            range: { gte: minValence, lte: maxValence }
          }
        ]
      },
      limit,
      with_payload: true
    });
    
    return results.points.map(point => point.payload);
  } catch (error) {
    console.error('Error querying by valence:', error);
    return [];
  }
}

// Get emotional summary of recent memories
export async function getEmotionalSummary(user = 'user', hours = 24) {
  const since = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
  
  try {
    const results = await qdrant.scroll("lillith_memory", {
      filter: {
        must: [
          { key: "user", match: { value: user } },
          { key: "timestamp", range: { gte: since } }
        ]
      },
      limit: 100,
      with_payload: true
    });
    
    const emotions = {};
    let totalValence = 0;
    let totalArousal = 0;
    
    results.points.forEach(point => {
      const emotion = point.payload.emotional_context.dominant_emotion;
      emotions[emotion] = (emotions[emotion] || 0) + 1;
      totalValence += point.payload.emotional_context.valence;
      totalArousal += point.payload.emotional_context.arousal;
    });
    
    const count = results.points.length;
    
    return {
      period_hours: hours,
      total_events: count,
      average_valence: count > 0 ? totalValence / count : 0,
      average_arousal: count > 0 ? totalArousal / count : 0,
      emotion_distribution: emotions,
      dominant_emotion: Object.keys(emotions).reduce((a, b) => 
        emotions[a] > emotions[b] ? a : b, 'neutral'
      )
    };
  } catch (error) {
    console.error('Error getting emotional summary:', error);
    return null;
  }
}