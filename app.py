"""
====================================================================
Plant Disease Detection - Flask Backend API
====================================================================
Author      : Senior AI Engineer
Description : REST API for plant disease detection using deep learning
Endpoints   : POST /predict, GET /health, GET /classes
====================================================================
"""

import os
import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FLASK APP INITIALIZATION
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)  # Enable cross-origin requests for API access

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    MODEL_PATH       = os.getenv("MODEL_PATH", "./model/plant_model.h5")
    LABELS_PATH      = os.getenv("LABELS_PATH", "./model/class_labels.json")
    IMG_SIZE         = (224, 224)
    MAX_FILE_SIZE    = 16 * 1024 * 1024   # 16MB
    ALLOWED_EXTS     = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    CONFIDENCE_THRESHOLD = 0.4            # Below this = "Uncertain"

app.config.from_object(Config)

# ─────────────────────────────────────────────
# DISEASE KNOWLEDGE BASE
# (In production: move to a database / external API)
# ─────────────────────────────────────────────
DISEASE_INFO = {
    # ── APPLE ────────────────────────────────────────────
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "description" : "Caused by Venturia inaequalis. Olive-green to black scab-like lesions on leaves and fruit. Major commercial apple disease worldwide.",
        "treatment"   : "Apply fungicides (captan, myclobutanil) from bud break through summer. Follow spray schedule based on local extension recommendations.",
        "prevention"  : "Plant scab-resistant varieties. Prune for good air circulation. Remove fallen infected leaves.",
    },
    "Apple___Black_rot": {
        "display_name": "Apple Black Rot",
        "description" : "Caused by Botryosphaeria obtusa. Purple spots on leaves, rotting fruit with brown rings, and cankers on branches.",
        "treatment"   : "Remove mummified fruit and infected limbs. Apply captan or thiophanate-methyl fungicides.",
        "prevention"  : "Remove dead wood and mummified fruit. Avoid wounding trees.",
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Apple Cedar Rust",
        "description" : "Caused by Gymnosporangium juniperi-virginianae. Bright orange-yellow spots on leaf upper surfaces.",
        "treatment"   : "Apply fungicides (myclobutanil, propiconazole) from pink stage through early June.",
        "prevention"  : "Remove nearby cedar/juniper trees. Plant rust-resistant apple varieties.",
    },
    "Apple___healthy": {
        "display_name": "Healthy Apple Plant",
        "description" : "Your apple tree looks healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed. Continue regular care.",
        "prevention"  : "Continue regular pruning, fertilization, and monitoring.",
    },

    # ── BLUEBERRY ─────────────────────────────────────────
    "Blueberry___healthy": {
        "display_name": "Healthy Blueberry Plant",
        "description" : "Your blueberry plant appears healthy! Continue good growing practices.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain acidic soil pH (4.5-5.5). Regular monitoring.",
    },

    # ── CHERRY ───────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry Powdery Mildew",
        "description" : "Caused by Podosphaera clandestina. White powdery coating on young leaves and shoots. Causes leaf curling and distortion.",
        "treatment"   : "Apply sulfur-based or potassium bicarbonate fungicides. Remove infected shoots.",
        "prevention"  : "Improve air circulation. Avoid overhead irrigation. Plant resistant varieties.",
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Healthy Cherry Plant",
        "description" : "Your cherry plant looks healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good orchard management.",
    },

    # ── CORN ─────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn Gray Leaf Spot",
        "description" : "Caused by Cercospora zeae-maydis. Rectangular gray-tan lesions running parallel to leaf veins.",
        "treatment"   : "Apply triazole or strobilurin fungicides. Remove severely infected leaves.",
        "prevention"  : "Plant resistant hybrids. Use crop rotation. Improve air circulation.",
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn Common Rust",
        "description" : "Caused by Puccinia sorghi. Small oval cinnamon-brown pustules on both leaf surfaces.",
        "treatment"   : "Apply fungicides (propiconazole or tebuconazole) at early rust development.",
        "prevention"  : "Plant resistant corn hybrids. Early planting helps avoid peak rust season.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display_name": "Corn Northern Leaf Blight",
        "description" : "Caused by Setosphaeria turcica. Large elliptical grayish-tan lesions on leaves. Can cause 30-50% yield loss.",
        "treatment"   : "Apply fungicides at tasseling stage. Most effective when disease is below ear level.",
        "prevention"  : "Plant resistant hybrids. Rotate with non-host crops. Till crop debris.",
    },
    "Corn_(maize)___healthy": {
        "display_name": "Healthy Corn Plant",
        "description" : "Your corn plant appears healthy! Normal green coloration with no disease symptoms.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain balanced fertilization. Scout regularly for early detection.",
    },

    # ── GRAPE ────────────────────────────────────────────
    "Grape___Black_rot": {
        "display_name": "Grape Black Rot",
        "description" : "Caused by Guignardia bidwellii. Brown spots on leaves and berries develop brown rot — eventually shriveling to black mummies.",
        "treatment"   : "Apply fungicides (captan, mancozeb) from bud break. Remove and destroy mummified berries.",
        "prevention"  : "Remove all mummified fruit. Prune for good air circulation.",
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape Esca (Black Measles)",
        "description" : "Complex disease caused by wood-rotting fungi. Tiger-stripe pattern on leaves with interveinal chlorosis.",
        "treatment"   : "No effective chemical cure. Remove infected wood during dormant pruning.",
        "prevention"  : "Make clean pruning cuts. Apply wound protectants immediately after pruning.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name": "Grape Leaf Blight",
        "description" : "Caused by Pseudocercospora vitis. Dark brown spots on leaf upper surfaces causing defoliation.",
        "treatment"   : "Apply copper-based fungicides. Remove infected leaves.",
        "prevention"  : "Maintain good canopy air circulation through proper pruning.",
    },
    "Grape___healthy": {
        "display_name": "Healthy Grape Vine",
        "description" : "Your grape vine appears healthy! Good color and no disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular canopy management and preventive fungicide program.",
    },

    # ── ORANGE ───────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display_name": "Citrus Greening (HLB)",
        "description" : "Most destructive citrus disease worldwide caused by Candidatus Liberibacter. Yellow shoots, blotchy leaves, misshapen bitter fruit. No cure exists.",
        "treatment"   : "No cure. Remove and destroy infected trees immediately. Control Asian citrus psyllid vector with insecticides.",
        "prevention"  : "Use certified disease-free nursery stock. Control psyllid populations. Inspect regularly.",
    },

    # ── PEACH ────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "display_name": "Peach Bacterial Spot",
        "description" : "Caused by Xanthomonas arboricola. Water-soaked spots on leaves turning brown with yellow halos. Sunken spots on fruit.",
        "treatment"   : "Apply copper bactericides during dormant season. Avoid overhead irrigation.",
        "prevention"  : "Plant resistant varieties. Proper pruning for air circulation.",
    },
    "Peach___healthy": {
        "display_name": "Healthy Peach Plant",
        "description" : "Your peach tree looks healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good orchard management practices.",
    },

    # ── PEPPER ───────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper Bacterial Spot",
        "description" : "Caused by Xanthomonas euvesicatoria. Water-soaked lesions turning brown/black on leaves. Raised scab-like spots on fruit.",
        "treatment"   : "Apply copper bactericides. Remove infected plant material.",
        "prevention"  : "Use disease-free seeds. Avoid overhead irrigation. Rotate crops.",
    },
    "Pepper,_bell___healthy": {
        "display_name": "Healthy Bell Pepper Plant",
        "description" : "Your bell pepper plant appears healthy! No disease symptoms visible.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good agricultural practices.",
    },

    # ── POTATO ───────────────────────────────────────────
    "Potato___Early_blight": {
        "display_name": "Potato Early Blight",
        "description" : "Caused by Alternaria solani. Dark brown concentric ring lesions on older leaves causing yellowing and defoliation.",
        "treatment"   : "Apply mancozeb or chlorothalonil fungicides. Remove infected leaves.",
        "prevention"  : "Use certified seed potatoes. Rotate crops. Avoid drought stress.",
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "description" : "Most devastating potato disease caused by Phytophthora infestans. Dark water-soaked lesions rapidly spreading — can destroy entire fields.",
        "treatment"   : "Emergency application of metalaxyl-M fungicides. Remove and destroy infected plants immediately.",
        "prevention"  : "Plant certified disease-free seed tubers. Use resistant varieties. Monitor blight forecasting services.",
    },
    "Potato___healthy": {
        "display_name": "Healthy Potato Plant",
        "description" : "Your potato plant looks healthy! The foliage shows normal green color without any disease symptoms.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring. Use certified seed potatoes for next season.",
    },

    # ── RASPBERRY ────────────────────────────────────────
    "Raspberry___healthy": {
        "display_name": "Healthy Raspberry Plant",
        "description" : "Your raspberry plant appears healthy! Continue good growing practices.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Regular pruning and monitoring for pests and diseases.",
    },

    # ── SOYBEAN ──────────────────────────────────────────
    "Soybean___healthy": {
        "display_name": "Healthy Soybean Plant",
        "description" : "Your soybean plant appears healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain crop rotation and proper field drainage.",
    },

    # ── SQUASH ───────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "display_name": "Squash Powdery Mildew",
        "description" : "Caused by Podosphaera xanthii. White powdery coating on leaves reducing photosynthesis and yield.",
        "treatment"   : "Apply sulfur-based fungicides or neem oil. Remove heavily infected leaves.",
        "prevention"  : "Improve air circulation. Avoid overhead watering. Plant resistant varieties.",
    },

    # ── STRAWBERRY ───────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry Leaf Scorch",
        "description" : "Caused by Diplocarpon earlianum. Small dark purple spots on leaves that enlarge and cause leaf margins to appear scorched.",
        "treatment"   : "Apply captan or thiram fungicides. Remove and destroy infected leaves.",
        "prevention"  : "Avoid overhead irrigation. Use certified disease-free plants. Rotate planting beds.",
    },
    "Strawberry___healthy": {
        "display_name": "Healthy Strawberry Plant",
        "description" : "Your strawberry plant appears healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and proper bed management.",
    },

    # ── TOMATO ───────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "description" : "Caused by Xanthomonas campestris. Small dark water-soaked spots on leaves and fruit with yellow halos.",
        "treatment"   : "Apply copper-based bactericides. Remove infected plant material immediately.",
        "prevention"  : "Use certified disease-free seeds. Avoid overhead irrigation. Rotate crops.",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "description" : "Caused by Alternaria solani. Dark brown spots with concentric rings (target-board pattern) on lower leaves first.",
        "treatment"   : "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
        "prevention"  : "Water at soil level. Space plants 45-60cm apart. Remove plant debris after harvest.",
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "description" : "Caused by Phytophthora infestans. Large dark water-soaked lesions on leaves. White fuzzy growth on undersides. Can destroy crop in days.",
        "treatment"   : "Immediately apply fungicides (cymoxanil + mancozeb). Remove and destroy infected plants.",
        "prevention"  : "Plant resistant varieties. Avoid overhead watering. Monitor weather forecasts.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "description" : "Caused by Cladosporium fulvum. Yellow patches on upper leaf with olive-green mold on undersides. Thrives in high humidity.",
        "treatment"   : "Improve ventilation. Apply chlorothalonil or copper fungicides.",
        "prevention"  : "Maintain humidity below 85%. Space plants properly.",
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "description" : "Caused by Septoria lycopersici. Small circular spots with dark borders and lighter centers on lower leaves.",
        "treatment"   : "Apply fungicides (chlorothalonil). Remove infected lower leaves. Mulch to prevent soil splash.",
        "prevention"  : "Rotate crops. Remove plant debris. Water at base of plant.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Spider Mites",
        "description" : "Tetranychus urticae infestation. Tiny mites cause stippled yellowing leaves. Fine webbing visible on undersides.",
        "treatment"   : "Spray strong water jets on undersides. Apply insecticidal soap or neem oil.",
        "prevention"  : "Maintain adequate soil moisture. Inspect plants regularly.",
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato Target Spot",
        "description" : "Caused by Corynespora cassiicola. Brown circular bullseye spots on leaves stems and fruit.",
        "treatment"   : "Apply fungicides (azoxystrobin or tebuconazole). Remove infected plant material.",
        "prevention"  : "Improve air circulation. Avoid leaf wetness. Use crop rotation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus",
        "description" : "Viral disease transmitted by whiteflies. Leaves curl upward and turn yellow with purple veins. Severe stunting. No cure.",
        "treatment"   : "No direct treatment. Remove infected plants immediately. Control whitefly populations.",
        "prevention"  : "Use whitefly-resistant varieties. Install insect-proof nets. Apply reflective mulches.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "description" : "Highly contagious RNA virus causing mottled light/dark green mosaic on leaves and distortion.",
        "treatment"   : "No cure. Remove infected plants. Disinfect all tools with 10% bleach solution.",
        "prevention"  : "Use certified virus-free seeds. Control aphids. Plant resistant varieties.",
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato Plant",
        "description" : "Your tomato plant appears healthy! Leaves show normal green coloration with no spots or abnormal patterns.",
        "treatment"   : "No treatment needed. Maintain regular watering and fertilization.",
        "prevention"  : "Continue current practices. Regularly inspect for early signs of disease.",
    },

    # ── DEFAULT ──────────────────────────────────────────
    "default": {
        "display_name": "Plant Disease Detected",
        "description" : "A plant disease has been detected. Please consult local agricultural extension services for specific diagnosis.",
        "treatment"   : "Consult a local agricultural expert for proper identification and treatment plan.",
        "prevention"  : "Practice good crop rotation, proper irrigation, and regular plant inspection.",
    }
}
# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
model = None
class_labels = {}

def load_model():
    """Load the trained TensorFlow model and class labels."""
    global model, class_labels

    model_path = app.config['MODEL_PATH']
    labels_path = app.config['LABELS_PATH']

    # Load class labels
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            # Keys are string indices from training
            raw = json.load(f)
            class_labels = {int(k): v for k, v in raw.items()}
        logger.info(f"✓ Loaded {len(class_labels)} class labels")
    else:
        logger.warning(f"Labels file not found: {labels_path}. Using disease info keys.")
        class_labels = {i: name for i, name in enumerate(DISEASE_INFO.keys())}

    # Load model
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Model loaded from {model_path}")
            logger.info(f"  Input shape : {model.input_shape}")
            logger.info(f"  Output shape: {model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning(f"Model file not found: {model_path}")
        logger.warning("Running in DEMO mode - predictions will be simulated")


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model inference.
    
    Pipeline:
    1. Decode bytes → PIL Image
    2. Convert to RGB (handle RGBA, grayscale)
    3. Resize to 224x224 (model input)
    4. OpenCV enhancement (CLAHE for better contrast)
    5. Normalize [0, 1]
    6. Add batch dimension
    
    Returns: numpy array of shape (1, 224, 224, 3)
    """
    # Load image from bytes
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (handles PNG with alpha, grayscale, etc.)
    pil_image = pil_image.convert('RGB')

    # Convert PIL → NumPy → OpenCV (for enhancement)
    img_np  = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Improves detection in varied lighting conditions
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    img_bgr  = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Resize to model input size
    img_size = app.config['IMG_SIZE']
    img_resized = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_AREA)

    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


def determine_severity(confidence):
    """
    Determine disease severity based on confidence score.
    
    Logic: Higher model confidence often correlates with
    more visible/advanced disease symptoms.
    """
    if confidence < 0.60:
        return "Mild"
    elif confidence < 0.85:
        return "Moderate"
    else:
        return "Severe"


def get_disease_info(class_name, confidence):
    """Get disease information from knowledge base."""
    info = DISEASE_INFO.get(class_name, DISEASE_INFO["default"])

    # Create a copy to avoid mutating the original
    result = {
        "disease_name"    : info["display_name"],
        "description"     : info["description"],
        "treatment"       : info["treatment"],
        "prevention"      : info["prevention"],
        "severity"        : determine_severity(confidence),
        "is_healthy"      : "healthy" in class_name.lower(),
    }
    return result


# ─────────────────────────────────────────────
# DEMO MODE (when model not available)
# ─────────────────────────────────────────────
def demo_prediction():
    """Return a simulated prediction for demo/testing purposes."""
    import random
    demo_classes = list(DISEASE_INFO.keys())
    demo_class   = random.choice(demo_classes)
    confidence   = round(random.uniform(0.75, 0.98), 4)
    return demo_class, confidence


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for deployment platforms."""
    return jsonify({
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.utcnow().isoformat(),
        "version"     : "1.0.0",
        "classes"     : len(class_labels)
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return list of supported disease classes."""
    return jsonify({
        "total"  : len(DISEASE_INFO),
        "classes": [info["display_name"] for key, info in DISEASE_INFO.items() if key != "default"]
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: image file
    
    Response (200 OK):
        {
            "disease_name"    : "Tomato Early Blight",
            "confidence_score": 0.9231,
            "description"     : "...",
            "treatment"       : "...",
            "prevention"      : "...",
            "severity"        : "Moderate",
            "is_healthy"      : false,
            "raw_class"       : "Tomato_Early_blight",
            "processing_time" : 0.234
        }
    
    Error Response (4xx/5xx):
        { "error": "Error message" }
    """
    start_time = time.time()
    logger.info(f"Prediction request received from {request.remote_addr}")

    # ── Validate request ──────────────────────────────────
    if 'image' not in request.files:
        logger.warning("No image in request")
        return jsonify({"error": "No image file provided. Include 'image' in form-data."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    # Validate file extension
    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    if ext not in app.config['ALLOWED_EXTS']:
        return jsonify({
            "error": f"Unsupported file type '.{ext}'. Allowed: {', '.join(app.config['ALLOWED_EXTS'])}"
        }), 400

    # Read image bytes
    image_bytes = file.read()

    # Validate file size
    if len(image_bytes) > app.config['MAX_FILE_SIZE']:
        return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file provided."}), 400

    # ── Preprocess ────────────────────────────────────────
    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 422

    # ── Inference ─────────────────────────────────────────
    try:
        if model is not None:
            # Real model prediction
            predictions = model.predict(img_array, verbose=0)
            class_idx   = int(np.argmax(predictions[0]))
            confidence  = float(predictions[0][class_idx])
            class_name  = class_labels.get(class_idx, "Unknown")

            # Top 3 predictions for debugging
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3 = [
                {
                    "class"     : class_labels.get(int(i), "Unknown"),
                    "confidence": float(predictions[0][i])
                }
                for i in top3_indices
            ]
        else:
            # Demo mode
            logger.warning("Model not loaded - using demo mode")
            class_name, confidence = demo_prediction()
            top3 = [{"class": class_name, "confidence": confidence}]

        # Low confidence threshold
        if confidence < app.config['CONFIDENCE_THRESHOLD']:
            return jsonify({
                "error"     : "Could not confidently identify disease. Please upload a clearer image of the plant leaf.",
                "confidence": round(confidence, 4)
            }), 200

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ── Build response ────────────────────────────────────
    disease_info = get_disease_info(class_name, confidence)
    processing_time = round(time.time() - start_time, 3)

    response = {
        **disease_info,
        "confidence_score" : round(confidence, 4),
        "raw_class"        : class_name,
        "processing_time"  : processing_time,
        "top_predictions"  : top3 if model is not None else [],
    }

    logger.info(
        f"Prediction: {class_name} | "
        f"Confidence: {confidence:.2%} | "
        f"Time: {processing_time}s"
    )

    return jsonify(response), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
with app.app_context():
    load_model()

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"

    logger.info(f"Starting Plant Disease Detection API on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Visit: http://localhost:{port}")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
