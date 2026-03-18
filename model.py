import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define global variables so app.py can access them
model = None
vectorizer = None

def train_model():
    global model, vectorizer
    print("Training model...")

    # Load the CLEANED file
    try:
        df = pd.read_csv("data/news_cleaned.csv")
    except FileNotFoundError:
        print("Error: data/news_cleaned.csv not found. Run clean_data.py first!")
        return

    if df.empty:
        raise ValueError("The cleaned dataset is empty. Check your CSV!")

    # Prepare features (X) and labels (y)
    X = df['text'].astype(str)
    # Map labels: 1 for REAL news, 0 for FAKE news
    y = df['label'].map({'REAL': 1, 'FAKE': 0})

    # Initialize vectorizer
    # min_df=5 ignores words that appear in fewer than 5 documents
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)
    
    print("Converting text to numbers (Vectorization)...")
    X_vec = vectorizer.fit_transform(X)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    print("Model trained successfully ✅")

def predict_news(text):
    global model, vectorizer
    if model is None or vectorizer is None:
        return "Model not trained"
        
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return "REAL" if pred == 1 else "FAKE"