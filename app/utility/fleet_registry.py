from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)
fleet_db = {}

@app.route('/beacon', methods=['POST'])
def beacon():
    data = request.json
    name = data.get('name')
    if not name:
        return {"error": "Missing ship name"}, 400
    fleet_db[name] = {
        "role": data.get("role", "unknown"),
        "status": data.get("status", "unknown"),
        "location": data.get("location", "unknown"),
        "last_seen": datetime.datetime.utcnow().isoformat(),
        "can_transform": data.get("can_transform", False),
        "signature": data.get("signature", "none")
    }
    return {"message": f"{name} registered successfully"}, 200

@app.route('/fleet', methods=['GET'])
def fleet():
    role = request.args.get('role')
    if role:
        filtered = {k: v for k, v in fleet_db.items() if v['role'] == role}
        return jsonify(filtered)
    return jsonify(fleet_db)

if __name__ == '__main__':
    app.run(debug=True)