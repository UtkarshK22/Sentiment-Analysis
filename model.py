import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def train_model():
    # Load dataset
    df = pd.read_csv("data/Amazon_Reviews.csv")

    # Select required columns
    df = df[['Review Text', 'Rating']].dropna()

    # Rename columns
    df.rename(columns={
        'Review Text': 'reviewText',
        'Rating': 'overall'
    }, inplace=True)

    # Convert rating → sentiment
    def convert_sentiment(rating):
        match = re.search(r'\d+', str(rating))
        if match:
            rating = int(match.group())
        else:
            return "neutral"

        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"

    # Apply sentiment
    df['sentiment'] = df['overall'].apply(convert_sentiment)

    # Clean text
    df['cleaned'] = df['reviewText'].apply(clean_text)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # Train model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Save model
    pickle.dump(model, open("model/model.pkl", "wb"))
    pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

    print("✅ Model trained and saved successfully!")

def predict_sentiment(text):
    model = pickle.load(open("model/model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]