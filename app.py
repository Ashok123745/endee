from flask import Flask, request
import model
from vector_db import search_similar

# Train model once (takes a moment)
print("Training model...")
model.train_model()

app = Flask(__name__)

# --- SHARED STYLES & ANIMATIONS ---
CSS_STYLE = """
<style>
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideRight {
        from { width: 0; }
        to { width: 100%; }
    }
    body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        background-color: #f0f2f5; 
        display: flex; 
        justify-content: center; 
        align-items: center; 
        min-height: 100vh; 
        margin: 0; 
    }
    .card { 
        background: white; 
        padding: 30px; 
        border-radius: 15px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
        width: 100%; 
        max-width: 600px; 
        animation: fadeInUp 0.7s ease-out; 
    }
    textarea { 
        width: 100%; 
        border: 1px solid #ddd; 
        border-radius: 8px; 
        padding: 10px; 
        font-size: 16px; 
    }
    .btn { 
        background: #007bff; 
        color: white; 
        border: none; 
        padding: 12px 20px; 
        border-radius: 8px; 
        cursor: pointer; 
        font-weight: bold; 
        transition: 0.3s;
    }
    .btn:hover { background: #0056b3; }
    .progress-bar { 
        background: #e9ecef; 
        height: 12px; 
        border-radius: 6px; 
        overflow: hidden; 
        margin: 20px 0; 
    }
    .progress-fill { 
        height: 100%; 
        border-radius: 6px; 
        animation: slideRight 1.5s ease-in-out forwards; 
    }
</style>
"""

# --- HOME PAGE ---
@app.route('/')
def home():
    return f'''
    {CSS_STYLE}
    <div class="card">
        <h2 style="color: #333;">🕵️ Fake News Detector</h2>
        <p style="color: #666;">Paste a news headline or article to verify its authenticity.</p>
        <form method="POST" action="/predict" onsubmit="document.getElementById('load').style.display='block'">
            <textarea name="news" rows="6" placeholder="Enter news text here..." required></textarea><br><br>
            <button type="submit" class="btn">Analyze News</button>
            <p id="load" style="display:none; color: #007bff; font-weight: bold; margin-top: 10px;">🔍 AI is thinking...</p>
        </form>
    </div>
    '''

# --- PREDICTION PAGE ---
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']

    # ML Prediction with Probability
    vector = model.vectorizer.transform([text])
    probabilities = model.model.predict_proba(vector)[0]
    prediction = model.model.predict(vector)[0]
    
    # Calculate Confidence
    confidence = max(probabilities) * 100
    result = "REAL" if prediction == 1 else "FAKE"
    color = "#28a745" if result == "REAL" else "#dc3545"

    # RAG Search
    similar = search_similar(text)

    return f'''
    {CSS_STYLE}
    <div class="card">
        <h2 style="margin-top: 0;">Result: <span style="color: {color};">{result}</span></h2>
        
        <div class="progress-bar">
            <div class="progress-fill" style="background: {color}; width: {confidence}%;"></div>
        </div>
        <p><strong>AI Confidence Score: {confidence:.2f}%</strong></p>
        
        <hr style="border: 0; border-top: 1px solid #eee;">
        
        <h4>Related Evidence from Database:</h4>
        <ul style="padding-left: 20px; color: #555; font-size: 14px;">
            <li style="margin-bottom: 10px;">{similar[0][:200]}...</li>
            <li style="margin-bottom: 10px;">{similar[1][:200]}...</li>
            <li style="margin-bottom: 10px;">{similar[2][:200]}...</li>
        </ul>
        
        <br>
        <a href="/" style="text-decoration: none; color: #007bff; font-weight: bold;">← Verify Another Article</a>
    </div>
    '''

if __name__ == '__main__':
    app.run(debug=False)