export async function logToLoki(message, labels = {}) {
  const payload = {
    streams: [{
      stream: { app: "lillith", ...labels },
      values: [[`${Date.now()}000000`, JSON.stringify(message)]]
    }]
  };

  try {
    await fetch("http://localhost:3100/loki/api/v1/push", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    console.error("Loki logging failed:", error);
  }
}

export async function logMessage(text, user, embeddingId, stage = "live") {
  await logToLoki(
    { text, embeddingId, timestamp: new Date().toISOString() },
    { user, stage, component: "chat" }
  );
}

export async function logSystemEvent(event, data = {}) {
  await logToLoki(
    { event, ...data, timestamp: new Date().toISOString() },
    { component: "system", level: "info" }
  );
}