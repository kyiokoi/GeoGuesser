"""
Flask web application for GeoGuesser country prediction
"""
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent / "src"))
from model import create_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL SELECTION: Choose which run to use
# ============================================================
# Options:
#   - "latest" : Use the most recent run (default)
#   - 1, 2, 3  : Use a specific run number
# Example: USE_RUN = 2  (to use models/Run 2)
USE_RUN = 4
# ============================================================

# Global variables
model = None
class_names = []
idx_to_class = {}


def load_latest_model():
    """Load the trained model based on USE_RUN configuration."""
    global model, class_names, idx_to_class
    
    # Find the Run directory
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No models directory found. Please train a model first.")
    
    run_dirs = [d for d in MODEL_PATH.iterdir() if d.is_dir() and d.name.startswith("Run ")]
    if not run_dirs:
        raise FileNotFoundError("No trained models found. Please run train.py first.")
    
    # Determine which run to use
    if USE_RUN == "latest":
        selected_run = max(run_dirs, key=lambda x: int(x.name.replace("Run ", "")))
        print(f"Using latest model (Run {selected_run.name.replace('Run ', '')})")
    else:
        # Use specific run number
        selected_run = MODEL_PATH / f"Run {USE_RUN}"
        if not selected_run.exists():
            available_runs = [d.name for d in run_dirs]
            raise FileNotFoundError(
                f"Run {USE_RUN} not found. Available runs: {', '.join(available_runs)}"
            )
        print(f"Using specified model (Run {USE_RUN})")
    
    print(f"Loading model from: {selected_run}")
    
    # Load class mapping
    class_mapping_path = selected_run / "class_to_idx.json"
    with open(class_mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    
    # Create reverse mapping (idx to class name)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    # Load model
    model = create_model(num_classes=len(class_names), pretrained=False)
    
    # Try to load best model, fallback to final model
    best_model_path = selected_run / "best_model.pt"
    final_model_path = selected_run / "final_model.pt"
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print("Loaded best_model.pt")
    elif final_model_path.exists():
        model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
        print("Loaded final_model.pt")
    else:
        raise FileNotFoundError(f"No model file found in {selected_run}")
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Device: {DEVICE}")
    
    return model, class_names, idx_to_class


def preprocess_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        image = Image.open(file.stream)
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidences = probabilities[0].cpu().numpy() * 100  # Convert to percentages
        
        # Get all predictions sorted by confidence
        predictions = []
        for idx, confidence in enumerate(confidences):
            predictions.append({
                'country': class_names[idx],
                'confidence': float(confidence)
            })
        
        # Sort by confidence (highest first)
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/model-info')
def model_info():
    """Return information about the loaded model."""
    try:
        # Find latest run
        run_dirs = [d for d in MODEL_PATH.iterdir() if d.is_dir() and d.name.startswith("Run ")]
        latest_run = max(run_dirs, key=lambda x: int(x.name.replace("Run ", "")))
        
        # Load run summary
        summary_path = latest_run / "run_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return jsonify(summary)
        else:
            return jsonify({'error': 'No model summary found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🌏 GeoGuesser Flask App Starting...")
    print("=" * 60)
    
    try:
        # Load model on startup
        load_latest_model()
        print("=" * 60)
        print("✓ Ready to accept requests!")
        print("=" * 60)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except Exception as e:
        print(f"❌ Error starting app: {str(e)}")
        print("Please make sure you have trained a model first by running: python src/train.py")
        sys.exit(1)
