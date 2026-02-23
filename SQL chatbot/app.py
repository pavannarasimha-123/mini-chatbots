from flask import Flask, request, jsonify, render_template
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    print("=== /ask API HIT ===")

    try:
        data = request.get_json()
        print("DATA RECEIVED:", data)

        if not data or "query" not in data:
            return jsonify({"result": "No query received"}), 400

        user_query = data["query"]
        print("USER QUERY:", user_query)

        reply = chatbot_response(user_query)
        print("BOT REPLY:", reply)

        return jsonify({"result": reply})

    except Exception as e:
        print("ERROR IN /ask:", e)
        return jsonify({"result": str(e)})

if __name__ == "__main__":
    app.run(debug=True)