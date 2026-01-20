import asyncio
import json
import websockets
from fastapi import FastAPI
from huggingface_hub import InferenceClient
import os

app = FastAPI()
llm = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=os.environ.get("HF_TOKEN", "your-hf-token"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
            memories = data["memories"]
            query = data["query"]
            context = data["context"]
            # Process three-loop recall
            response_text = ""
            for loop in memories[0] if memories else []:
                response_text += f"{loop['weight']}: {loop['content']}\nEmotion: {json.dumps(loop['emotion'])}\n"
                if 'judgment' in loop:
                    response_text += f"Judgment: {json.dumps(loop['judgment'])}\n"
                response_text += "---\n"
            prompt = f"Query: {query}\nContext: {json.dumps(context)}\nRecall Output: {response_text}\nGenerate a response integrating emotional context and logical analysis."
            response = llm.text_generation(prompt, max_new_tokens=200)
            emotion = memories[0][0]["emotion"] if memories else {"speed": "normal", "energy": "medium"}
            await websocket.send_json({"response": response, "emotion": emotion})
        except:
            await websocket.send_json({"error": "Processing failed"})
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)