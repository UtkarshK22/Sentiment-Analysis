from flask import Flask, render_template, request
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Step 1: Simple dataset
data = {
    "text": [
        "Absolutely fantastic",
        "Not good at all",
        "Really enjoyed it",
        "Very happy with service",
        "This is amazing",
        "Worst experience ever",
        "I hate this",
        "This is terrible",
        "Very bad quality"
        "I love this product",
    ],
    "label": ["positive", "positive", "positive", "negative", "negative", "negative", "positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df["text"] = df["text"].apply(clean_text)
df["label"] = df["label"].map({"positive": 1, "negative": 0})

# Step 3: TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

# Step 4: Model
model = LogisticRegression()
model.fit(X, df["label"])

# Step 5: Prediction function
def predict_sentiment(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)

    return "POSITIVE 😊" if result[0] == 1 else "NEGATIVE 😠"

# Step 6: Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        user_text = request.form["text"]
        prediction = predict_sentiment(user_text)

    return render_template("index.html", prediction=prediction)

# Step 7: Run app
if __name__ == "__main__":
    app.run(debug=True)