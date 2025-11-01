from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from arguments import args
from analyzer import Analyzer


app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

analyzer = Analyzer(will_train=False, args=args)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    sentiment, percentage = analyzer.classify_sentiment(text)
    result = {}
    result["sentiment"] = sentiment
    result["percentage"] = percentage
    return render_template("index.html", result=result)
 

if __name__ == "__main__":
    app.run(debug=True)
