from flask import Blueprint, jsonify, request

vote_bp = Blueprint("vote", __name__)

@vote_bp.route("/status", methods=["GET"])
def vote_status():
    return jsonify({"status": "Council vote system online."})

@vote_bp.route("/cast", methods=["POST"])
def cast_vote():
    data = request.get_json()
    decision = data.get("decision", "abstain")
    return jsonify({"message": f"Vote registered: {decision}"})
