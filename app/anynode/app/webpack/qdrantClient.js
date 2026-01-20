import { QdrantClient } from "qdrant-node";

export const qdrant = new QdrantClient({
  url: "http://localhost:6333"
});

export async function initializeCollections() {
  try {
    await qdrant.createCollection("lillith_messages", {
      vectors: { size: 1536, distance: "Cosine" }
    });
    await qdrant.createCollection("lillith_archive", {
      vectors: { size: 1536, distance: "Cosine" }
    });
  } catch (error) {
    console.log("Collections exist:", error.message);
  }
}

export async function storeMessage(text, user, embedding) {
  const id = Date.now();
  await qdrant.upsert("lillith_messages", {
    points: [{
      id,
      vector: embedding,
      payload: { text, user, timestamp: new Date().toISOString(), stage: "live" }
    }]
  });
  return id;
}

export async function findSimilarMessages(embedding, limit = 5) {
  const results = await qdrant.search("lillith_messages", {
    vector: embedding,
    limit,
    with_payload: true
  });
  return results.map(match => ({
    text: match.payload.text,
    user: match.payload.user,
    score: match.score,
    timestamp: match.payload.timestamp
  }));
}