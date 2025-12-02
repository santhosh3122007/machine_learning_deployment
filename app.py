from flask import Flask, request, jsonify
import re
import joblib
import os

app = Flask(__name__)

# Load the SVC model and TF-IDF vectorizer
svc_model = joblib.load('svc_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
print("SVC model and TF-IDF vectorizer loaded successfully.")

# Define the remove_stopwords function
def remove_stopwords(text):
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'SVC Model API is running', 
        'status': 'ok',
        'endpoints': {
            'predict': '/predict [POST]'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'text' not in data:
            return jsonify({'error': 'Missing text in request body'}), 400
        
        input_text = data['text']
        
        # Preprocess the input text
        processed_text = str(input_text).lower()
        processed_text = re.sub(r'[^a-zA-Z0-9\s]', '', processed_text)
        processed_text = remove_stopwords(processed_text)
        
        # Vectorize the preprocessed text
        text_vectorized = tfidf_vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = svc_model.predict(text_vectorized)[0]
        
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("'/predict' endpoint defined successfully.")

if __name__ == "__main__":
    # CRITICAL: Use 0.0.0.0 and PORT from environment for Render
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
