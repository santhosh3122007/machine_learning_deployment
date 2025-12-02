from flask import Flask, request, jsonify
import re
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
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

    return jsonify({'prediction': prediction})

print("'/predict' endpoint defined successfully.")

if __name__ == "__main__":
    app.run(debug=True)
