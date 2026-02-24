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
    # ── TOMATO ──────────────────────────────────────────
    "Tomato_Bacterial_spot": {
        "display_name" : "Tomato Bacterial Spot",
        "description"  : "Caused by Xanthomonas campestris pv. vesicatoria. Small, dark, water-soaked spots appear on leaves, stems, and fruit. Spots may have yellow halos and eventually turn brown and necrotic.",
        "treatment"    : "Apply copper-based bactericides (copper hydroxide or copper oxychloride). Remove infected plant material immediately. Use streptomycin sulfate sprays in severe cases. Ensure proper plant spacing for air circulation.",
        "prevention"   : "Use certified disease-free seeds. Avoid overhead irrigation. Rotate crops every 2-3 years. Disinfect gardening tools regularly. Plant resistant varieties when available.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Early_blight": {
        "display_name" : "Tomato Early Blight",
        "description"  : "Caused by Alternaria solani. Dark brown spots with concentric rings (target-board pattern) appear on lower leaves first, then spread upward. Infected leaves turn yellow and drop.",
        "treatment"    : "Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin. Remove infected leaves. Mulch soil to reduce spore splash. Maintain adequate potassium levels.",
        "prevention"   : "Water at soil level, not on leaves. Space plants 45-60cm apart. Remove plant debris after harvest. Use crop rotation with non-Solanaceous crops.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Late_blight": {
        "display_name" : "Tomato Late Blight",
        "description"  : "Caused by Phytophthora infestans (same pathogen as Irish Potato Famine). Large, dark, water-soaked lesions on leaves. White fuzzy growth visible on undersides. Can destroy entire crop in days.",
        "treatment"    : "Immediately apply fungicides (cymoxanil + mancozeb, or metalaxyl). Remove and destroy infected plants. Do not compost infected material. Apply systemic fungicides if disease is established.",
        "prevention"   : "Plant resistant varieties (Mountain Magic, Defiant). Avoid overhead watering. Ensure good air circulation. Monitor weather forecasts for blight conditions (cool, wet weather).",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Leaf_Mold": {
        "display_name" : "Tomato Leaf Mold",
        "description"  : "Caused by Cladosporium fulvum. Yellow patches on upper leaf surface with olive-green to grayish-purple mold on undersides. Thrives in high humidity (>85%) greenhouse conditions.",
        "treatment"    : "Improve greenhouse ventilation. Apply fungicides (chlorothalonil, mancozeb, or copper fungicides). Remove and destroy heavily infected leaves.",
        "prevention"   : "Maintain humidity below 85%. Space plants properly. Use resistant varieties. Avoid wetting foliage when watering.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name" : "Tomato Septoria Leaf Spot",
        "description"  : "Caused by Septoria lycopersici. Small circular spots (3-6mm) with dark brown borders and lighter centers appear on lower leaves. Tiny black specks (pycnidia) visible in spot centers.",
        "treatment"    : "Apply fungicides (chlorothalonil, copper fungicides). Remove infected lower leaves. Mulch to prevent soil splash.",
        "prevention"   : "Rotate crops. Remove plant debris. Water at base. Stake plants to improve airflow.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display_name" : "Two-Spotted Spider Mite",
        "description"  : "Tetranychus urticae infestation. Tiny mites (not visible to naked eye) cause stippled, yellowing leaves. Fine webbing visible on undersides. Thrives in hot, dry conditions.",
        "treatment"    : "Spray strong water jets on plant undersides. Apply miticides (abamectin, bifenazate). Use insecticidal soap or neem oil sprays. Introduce predatory mites (Phytoseiulus persimilis).",
        "prevention"   : "Maintain adequate soil moisture. Avoid dusty conditions. Inspect plants regularly. Avoid broad-spectrum insecticides that kill natural predators.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Target_Spot": {
        "display_name" : "Tomato Target Spot",
        "description"  : "Caused by Corynespora cassiicola. Brown circular spots with concentric rings resembling a bullseye. Spots may coalesce causing large dead areas. Affects leaves, stems, and fruit.",
        "treatment"    : "Apply fungicides (azoxystrobin, pyraclostrobin, or tebuconazole). Remove infected plant material. Ensure proper field sanitation.",
        "prevention"   : "Improve air circulation. Avoid leaf wetness. Use crop rotation. Remove and burn infected debris after harvest.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name" : "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "description"  : "Viral disease transmitted by whiteflies (Bemisia tabaci). Leaves curl upward and inward, turn yellow with purple veins. Severe stunting. No cure once infected.",
        "treatment"    : "No direct treatment. Remove and destroy infected plants immediately to prevent spread. Control whitefly populations with insecticides or yellow sticky traps.",
        "prevention"   : "Use whitefly-resistant varieties. Install insect-proof nets. Apply reflective mulches to repel whiteflies. Use systemic insecticides for whitefly control before planting.",
        "severity_map" : {"low": "Moderate", "medium": "Severe", "high": "Severe"}
    },
    "Tomato_Tomato_mosaic_virus": {
        "display_name" : "Tomato Mosaic Virus (ToMV)",
        "description"  : "Highly contagious RNA virus causing mottled light/dark green mosaic on leaves, distortion, and reduced fruit quality. Spreads through contact, contaminated tools, and seeds.",
        "treatment"    : "No cure. Remove infected plants. Disinfect all tools with 10% bleach solution. Wash hands thoroughly before handling plants.",
        "prevention"   : "Use certified virus-free seeds. Avoid tobacco use near plants (TMV related). Plant resistant varieties. Control aphids and other insect vectors.",
        "severity_map" : {"low": "Moderate", "medium": "Severe", "high": "Severe"}
    },
    "Tomato_healthy": {
        "display_name" : "Healthy Tomato Plant",
        "description"  : "Your tomato plant appears healthy! Leaves show normal green coloration, no spots, discoloration, or abnormal patterns. Continue good growing practices to maintain plant health.",
        "treatment"    : "No treatment needed. Maintain regular watering and fertilization schedule.",
        "prevention"   : "Continue current practices. Regularly inspect for early signs of disease or pest. Maintain proper plant spacing and good air circulation.",
        "severity_map" : {}
    },

    # ── POTATO ──────────────────────────────────────────
    "Potato_Early_blight": {
        "display_name" : "Potato Early Blight",
        "description"  : "Caused by Alternaria solani. Dark brown concentric ring lesions on older leaves. Yellowing around lesions. Can cause significant defoliation and tuber quality reduction.",
        "treatment"    : "Apply mancozeb, chlorothalonil, or azoxystrobin fungicides. Remove infected leaves. Ensure balanced nutrition (adequate potassium).",
        "prevention"   : "Use certified seed potatoes. Rotate crops. Avoid drought stress. Apply preventive fungicides in high-risk periods.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Potato_Late_blight": {
        "display_name" : "Potato Late Blight",
        "description"  : "The most devastating potato disease, caused by Phytophthora infestans. Dark water-soaked lesions rapidly spreading on leaves and stems. White sporulation visible in moist conditions. Can destroy entire fields.",
        "treatment"    : "Emergency application of metalaxyl-M or cymoxanil fungicides. Remove and destroy infected plant material. Do not allow infected tubers to remain in soil.",
        "prevention"   : "Plant certified disease-free seed tubers. Use resistant varieties (Sarpo Mira, Cara). Apply preventive fungicides when weather conditions favor disease. Monitor blight forecasting services.",
        "severity_map" : {"low": "Moderate", "medium": "Severe", "high": "Severe"}
    },
    "Potato_healthy": {
        "display_name" : "Healthy Potato Plant",
        "description"  : "Your potato plant looks healthy! The foliage shows normal green color without any disease symptoms. Continue good agricultural practices.",
        "treatment"    : "No treatment needed.",
        "prevention"   : "Continue regular monitoring. Use certified seed potatoes for next season. Practice crop rotation.",
        "severity_map" : {}
    },

    # ── CORN (MAIZE) ────────────────────────────────────
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name" : "Corn Gray Leaf Spot",
        "description"  : "Caused by Cercospora zeae-maydis. Rectangular gray-tan lesions running parallel to leaf veins. Lesions may have yellow or brown borders. Severe cases cause significant leaf blighting.",
        "treatment"    : "Apply triazole or strobilurin fungicides. Remove severely infected leaves. Ensure proper plant nutrition.",
        "prevention"   : "Plant resistant hybrids. Use crop rotation (avoid continuous corn). Till infected crop debris. Improve air circulation with proper plant spacing.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Corn_(maize)_Common_rust_": {
        "display_name" : "Corn Common Rust",
        "description"  : "Caused by Puccinia sorghi. Small, oval to elongated, cinnamon-brown pustules on both leaf surfaces. Severe infections cause premature leaf death and reduced photosynthesis.",
        "treatment"    : "Apply fungicides (triazoles like propiconazole or tebuconazole) at early rust development. Most effective when applied before 50% of leaves are infected.",
        "prevention"   : "Plant resistant corn hybrids. Early planting can help avoid peak rust season. Fungicide application at tassel stage in high-risk areas.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Corn_(maize)_Northern_Leaf_Blight": {
        "display_name" : "Corn Northern Leaf Blight",
        "description"  : "Caused by Setosphaeria turcica. Large, elliptical, grayish-tan lesions (2.5-15cm) on leaves. Infected leaves die prematurely. Can cause 30-50% yield loss in severe cases.",
        "treatment"    : "Apply fungicides (strobilurins, triazoles, or their combinations) at tasseling. Applications most effective when disease is below ear level.",
        "prevention"   : "Plant resistant hybrids. Rotate with non-host crops. Till crop debris. Avoid excessive nitrogen fertilization.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Corn_(maize)_healthy": {
        "display_name" : "Healthy Corn Plant",
        "description"  : "Your corn plant appears to be in excellent health! Normal green coloration with no disease symptoms detected.",
        "treatment"    : "No treatment needed. Continue regular care.",
        "prevention"   : "Maintain balanced fertilization. Scout regularly for early pest and disease detection.",
        "severity_map" : {}
    },

    # ── APPLE ────────────────────────────────────────────
    "Apple_Apple_scab": {
        "display_name" : "Apple Scab",
        "description"  : "Caused by Venturia inaequalis. Olive-green to black scab-like lesions on leaves, fruit, and twigs. Severe infections cause leaf drop and fruit cracking. Major commercial apple disease worldwide.",
        "treatment"    : "Apply fungicides (captan, myclobutanil, or trifloxystrobin) from bud break through summer. Follow spray schedule based on local extension recommendations.",
        "prevention"   : "Plant scab-resistant varieties (Liberty, Enterprise, GoldRush). Prune for good air circulation. Remove fallen infected leaves. Apply dormant copper sprays.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Apple_Black_rot": {
        "display_name" : "Apple Black Rot",
        "description"  : "Caused by Botryosphaeria obtusa. Purple spots on leaves, rotting fruit with brown rings, and cankers on branches. Infected fruit mummifies and remains on tree spreading infection.",
        "treatment"    : "Remove mummified fruit and infected limbs. Apply captan or thiophanate-methyl fungicides. Prune cankers to remove infected wood.",
        "prevention"   : "Remove dead wood and mummified fruit. Avoid wounding trees. Maintain tree vigor with proper nutrition and irrigation.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Apple_Cedar_apple_rust": {
        "display_name" : "Apple Cedar Rust",
        "description"  : "Caused by Gymnosporangium juniperi-virginianae. Bright orange-yellow spots on leaf upper surfaces. Requires both apple and cedar/juniper hosts to complete life cycle.",
        "treatment"    : "Apply fungicides (myclobutanil, propiconazole) from pink stage through early June. Multiple applications needed.",
        "prevention"   : "Remove nearby cedar/juniper trees if possible. Plant rust-resistant apple varieties. Apply preventive fungicide sprays.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Apple_healthy": {
        "display_name" : "Healthy Apple Plant",
        "description"  : "Your apple tree looks healthy! No disease symptoms detected. Maintain good orchard management practices.",
        "treatment"    : "No treatment needed.",
        "prevention"   : "Continue regular pruning, fertilization, and monitoring. Maintain proper spacing for air circulation.",
        "severity_map" : {}
    },

    # ── GRAPE ────────────────────────────────────────────
    "Grape_Black_rot": {
        "display_name" : "Grape Black Rot",
        "description"  : "Caused by Guignardia bidwellii. Small yellow-brown spots on leaves with black borders. Berries develop brown rot and eventually shrivel to hard black mummies. Can cause total crop loss.",
        "treatment"    : "Apply fungicides (captan, mancozeb, myclobutanil) from bud break. Remove and destroy mummified berries. Prune to open canopy.",
        "prevention"   : "Remove all mummified fruit. Prune for good air circulation. Apply fungicide spray program from early season. Use resistant grape varieties.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Grape_Esca_(Black_Measles)": {
        "display_name" : "Grape Esca (Black Measles)",
        "description"  : "Complex disease caused by multiple wood-rotting fungi. Tiger-stripe pattern on leaves (interveinal chlorosis/necrosis). Berries develop dark spots. Chronic form causes gradual vine decline.",
        "treatment"    : "No effective chemical cure. Remove infected wood during dormant pruning. Protect pruning wounds with wound sealants or fungicides (tebuconazole).",
        "prevention"   : "Make clean pruning cuts. Apply wound protectants immediately after pruning. Avoid large pruning wounds. Remove and destroy infected wood.",
        "severity_map" : {"low": "Moderate", "medium": "Severe", "high": "Severe"}
    },
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name" : "Grape Leaf Blight",
        "description"  : "Caused by Pseudocercospora vitis. Dark brown spots on leaf upper surfaces. Severe infections cause defoliation, weakening vines and reducing fruit quality.",
        "treatment"    : "Apply copper-based or organic fungicides. Remove infected leaves. Improve air circulation through canopy management.",
        "prevention"   : "Maintain good canopy air circulation through proper pruning. Remove fallen infected leaves. Apply preventive fungicide sprays during wet seasons.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Grape_healthy": {
        "display_name" : "Healthy Grape Vine",
        "description"  : "Your grape vine appears healthy! Good color and no disease symptoms detected. Continue vineyard management best practices.",
        "treatment"    : "No treatment needed.",
        "prevention"   : "Continue regular canopy management. Apply preventive fungicide program during growing season.",
        "severity_map" : {}
    },

    # ── PEPPER ───────────────────────────────────────────
    "Pepper,_bell_Bacterial_spot": {
        "display_name" : "Bell Pepper Bacterial Spot",
        "description"  : "Caused by Xanthomonas euvesicatoria. Water-soaked lesions on leaves that turn brown/black. Raised scab-like spots on fruit. Severely reduces marketable yield.",
        "treatment"    : "Apply copper bactericides. Remove infected plant material. Avoid working in fields when plants are wet.",
        "prevention"   : "Use disease-free seeds/transplants. Avoid overhead irrigation. Rotate crops. Use resistant varieties.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
    },
    "Pepper,_bell_healthy": {
        "display_name" : "Healthy Bell Pepper Plant",
        "description"  : "Your bell pepper plant appears healthy! No disease symptoms visible. Maintain current growing practices.",
        "treatment"    : "No treatment needed.",
        "prevention"   : "Continue regular monitoring and good agricultural practices.",
        "severity_map" : {}
    },

    # ── DEFAULT FALLBACK ─────────────────────────────────
    "default": {
        "display_name" : "Plant Disease Detected",
        "description"  : "A plant disease has been detected. Please consult local agricultural extension services for specific diagnosis and treatment recommendations.",
        "treatment"    : "Consult a local agricultural expert or extension service for proper identification and treatment plan.",
        "prevention"   : "Practice good crop rotation, proper irrigation, and regular plant inspection to prevent disease spread.",
        "severity_map" : {"low": "Mild", "medium": "Moderate", "high": "Severe"}
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
