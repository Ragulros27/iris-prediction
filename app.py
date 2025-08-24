from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import os

app = Flask(__name__)

def load_model():
    model_path = 'model/model.pkl'
    
    if not os.path.exists(model_path):
        print("Training new model...")
        
        # Load sample Iris dataset
        data = load_iris()
        X, y = data.data, data.target
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save the model
        joblib.dump(model, model_path)
        print(f"Model trained and saved at {model_path}!")
        
        # Save class names for reference
        class_names = data.target_names
        joblib.dump(class_names, 'model/class_names.pkl')
        print("Class names saved!")
    
    return joblib.load(model_path)

# Load model when application starts
model = load_model()
class_names = joblib.load('model/class_names.pkl') if os.path.exists('model/class_names.pkl') else ['Class 0', 'Class 1', 'Class 2']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        if features.shape[1] != 4:
            return jsonify({'error': f'Expected 4 features, got {features.shape[1]}'}), 400
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'class_name': class_names[prediction[0]],
            'probability': probability[0].tolist(),
            'confidence': float(np.max(probability[0])),
            'message': 'Success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': True,
        'model_type': 'RandomForestClassifier',
        'classes': class_names.tolist() if hasattr(class_names, 'tolist') else class_names
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)