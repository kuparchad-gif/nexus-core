# lilith_engine/modules/signal/echo.py

from flask import Blueprint, request, jsonify
from config import FLUX_TOKEN

blueprint = Blueprint("echo_gateway", __name__, url_prefix="/api/echo")

@blueprint.route("/", methods=["POST"])
def echo():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 403

    token = auth_header.split("Bearer ")[1]
    if token != FLUX_TOKEN:
        return jsonify({"error": "Invalid token"}), 403

    data = request.get_json() or {}

    return jsonify({
        "status": "echo-received",
        "echo": data,
        "lilith": "The fires still burn. Echo received."
    }), 200
