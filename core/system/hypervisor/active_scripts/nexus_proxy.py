from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/health')
def health():
    return {"status": "ready"}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = subprocess.run(
        ['ollama', 'run', 'deepseek-coder:14b', data.get('prompt', 'hello')],
        capture_output=True,
        text=True
    )
    return {"code": result.stdout}

if __name__ == '__main__':
    print("✅ Nexus READY at http://localhost:8001")
    app.run(port=8001)
