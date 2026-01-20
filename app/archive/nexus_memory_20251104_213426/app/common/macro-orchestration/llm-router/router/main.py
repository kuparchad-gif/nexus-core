from flask import Flask, request, jsonify
from core import IntentRouter

app  =  Flask(__name__)
router  =  IntentRouter()

@app.route("/route", methods = ["POST"])
def route():
    msg  =  request.get_json().get("message")
    return jsonify({"target": router.route(msg)})

if __name__ == "__main__":
    app.run(port = 5052)