from urllib import response
from dotenv import load_dotenv
import os
import io
import json
import logging
import time
import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

class Config:
    MODEL_PATH           = os.getenv("MODEL_PATH", str(BASE_DIR / "model" / "plant_model.h5"))
    LABELS_PATH          = os.getenv("LABELS_PATH", str(BASE_DIR / "model" / "class_labels.json"))
    IMG_SIZE             = (224, 224)
    MAX_FILE_SIZE        = 16 * 1024 * 1024
    ALLOWED_EXTS         = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
    CONFIDENCE_THRESHOLD = 0.4
    USE_CLAHE_PREPROCESS = os.getenv("USE_CLAHE_PREPROCESS", "false").strip().lower() in {"1", "true", "yes", "on"}

app.config.from_object(Config)

def _resolve_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()

def _get_gemini_api_key():
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )

def get_ai_description(disease_name, confidence, is_healthy, lang='en'):
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel("gemini-2.0-flash")
        lang_instruction = "Reply in Hindi only." if lang == 'hi' else "Reply in English only."
        if is_healthy:
            prompt = f"""{lang_instruction}
A farmer's plant image was analyzed.
Result: {disease_name} - Plant is HEALTHY (Confidence: {confidence:.1%})
Reply in this EXACT JSON format only:
{{
    "description": "2-3 simple sentences about healthy plant appearance and what it means for the farmer",
    "treatment": "2-3 sentences about maintaining plant health, watering, fertilization tips",
    "prevention": "2-3 sentences about preventing future diseases for this specific plant"
}}
Use simple farmer-friendly language. Return only JSON, nothing else."""
        else:
            prompt = f"""{lang_instruction}
A farmer's plant leaf image was analyzed by AI.
Detected Disease: {disease_name}
Confidence: {confidence:.1%}
Reply in this EXACT JSON format only:
{{
    "description": "3-4 simple sentences: what disease is this, what causes it, what symptoms appear, how serious is it",
    "treatment": "3-4 simple sentences: exact steps to treat, which products to use, how to apply them",
    "prevention": "3-4 simple sentences: how to prevent this disease next time, best practices for this crop"
}}
Use simple farmer-friendly language. Be specific and practical. Return only JSON, nothing else."""
        response = gemini.generate_content(prompt)
        response_text = response.text.strip()
        response_text = re.sub(r'```json|```', '', response_text).strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            ai_data = json.loads(json_match.group())
            logger.info(f"Gemini response for: {disease_name}")
            return ai_data
    except Exception as e:
        logger.warning(f"Gemini API failed: {e} - using local info")
        return None

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "description" : "Caused by Venturia inaequalis. Olive-green to black scab-like lesions on leaves and fruit. Major commercial apple disease worldwide.",
        "treatment"   : "Apply fungicides (captan, myclobutanil) from bud break through summer.",
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
    "Blueberry___healthy": {
        "display_name": "Healthy Blueberry Plant",
        "description" : "Your blueberry plant appears healthy! Continue good growing practices.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain acidic soil pH (4.5-5.5). Regular monitoring.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry Powdery Mildew",
        "description" : "Caused by Podosphaera clandestina. White powdery coating on young leaves and shoots.",
        "treatment"   : "Apply sulfur-based or potassium bicarbonate fungicides. Remove infected shoots.",
        "prevention"  : "Improve air circulation. Avoid overhead irrigation. Plant resistant varieties.",
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Healthy Cherry Plant",
        "description" : "Your cherry plant looks healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good orchard management.",
    },
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
        "description" : "Caused by Setosphaeria turcica. Large elliptical grayish-tan lesions on leaves.",
        "treatment"   : "Apply fungicides at tasseling stage. Most effective when disease is below ear level.",
        "prevention"  : "Plant resistant hybrids. Rotate with non-host crops. Till crop debris.",
    },
    "Corn_(maize)___healthy": {
        "display_name": "Healthy Corn Plant",
        "description" : "Your corn plant appears healthy! Normal green coloration with no disease symptoms.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain balanced fertilization. Scout regularly for early detection.",
    },
    "Grape___Black_rot": {
        "display_name": "Grape Black Rot",
        "description" : "Caused by Guignardia bidwellii. Brown spots on leaves and berries develop brown rot.",
        "treatment"   : "Apply fungicides (captan, mancozeb) from bud break. Remove and destroy mummified berries.",
        "prevention"  : "Remove all mummified fruit. Prune for good air circulation.",
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape Esca (Black Measles)",
        "description" : "Complex disease caused by wood-rotting fungi. Tiger-stripe pattern on leaves.",
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
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display_name": "Citrus Greening (HLB)",
        "description" : "Most destructive citrus disease worldwide. Yellow shoots, blotchy leaves, misshapen bitter fruit. No cure exists.",
        "treatment"   : "No cure. Remove and destroy infected trees immediately. Control Asian citrus psyllid vector.",
        "prevention"  : "Use certified disease-free nursery stock. Control psyllid populations.",
    },
    "Peach___Bacterial_spot": {
        "display_name": "Peach Bacterial Spot",
        "description" : "Caused by Xanthomonas arboricola. Water-soaked spots on leaves turning brown with yellow halos.",
        "treatment"   : "Apply copper bactericides during dormant season. Avoid overhead irrigation.",
        "prevention"  : "Plant resistant varieties. Proper pruning for air circulation.",
    },
    "Peach___healthy": {
        "display_name": "Healthy Peach Plant",
        "description" : "Your peach tree looks healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good orchard management practices.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper Bacterial Spot",
        "description" : "Caused by Xanthomonas euvesicatoria. Water-soaked lesions turning brown/black on leaves.",
        "treatment"   : "Apply copper bactericides. Remove infected plant material.",
        "prevention"  : "Use disease-free seeds. Avoid overhead irrigation. Rotate crops.",
    },
    "Pepper,_bell___healthy": {
        "display_name": "Healthy Bell Pepper Plant",
        "description" : "Your bell pepper plant appears healthy! No disease symptoms visible.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and good agricultural practices.",
    },
    "Potato___Early_blight": {
        "display_name": "Potato Early Blight",
        "description" : "Caused by Alternaria solani. Dark brown concentric ring lesions on older leaves.",
        "treatment"   : "Apply mancozeb or chlorothalonil fungicides. Remove infected leaves.",
        "prevention"  : "Use certified seed potatoes. Rotate crops. Avoid drought stress.",
    },
    "Potato___Late_blight": {
        "display_name": "Potato Late Blight",
        "description" : "Most devastating potato disease caused by Phytophthora infestans. Dark water-soaked lesions rapidly spreading.",
        "treatment"   : "Emergency application of metalaxyl-M fungicides. Remove and destroy infected plants immediately.",
        "prevention"  : "Plant certified disease-free seed tubers. Use resistant varieties.",
    },
    "Potato___healthy": {
        "display_name": "Healthy Potato Plant",
        "description" : "Your potato plant looks healthy! The foliage shows normal green color.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring. Use certified seed potatoes for next season.",
    },
    "Raspberry___healthy": {
        "display_name": "Healthy Raspberry Plant",
        "description" : "Your raspberry plant appears healthy! Continue good growing practices.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Regular pruning and monitoring for pests and diseases.",
    },
    "Soybean___healthy": {
        "display_name": "Healthy Soybean Plant",
        "description" : "Your soybean plant appears healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Maintain crop rotation and proper field drainage.",
    },
    "Squash___Powdery_mildew": {
        "display_name": "Squash Powdery Mildew",
        "description" : "Caused by Podosphaera xanthii. White powdery coating on leaves reducing photosynthesis.",
        "treatment"   : "Apply sulfur-based fungicides or neem oil. Remove heavily infected leaves.",
        "prevention"  : "Improve air circulation. Avoid overhead watering. Plant resistant varieties.",
    },
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry Leaf Scorch",
        "description" : "Caused by Diplocarpon earlianum. Small dark purple spots on leaves that enlarge.",
        "treatment"   : "Apply captan or thiram fungicides. Remove and destroy infected leaves.",
        "prevention"  : "Avoid overhead irrigation. Use certified disease-free plants.",
    },
    "Strawberry___healthy": {
        "display_name": "Healthy Strawberry Plant",
        "description" : "Your strawberry plant appears healthy! No disease symptoms detected.",
        "treatment"   : "No treatment needed.",
        "prevention"  : "Continue regular monitoring and proper bed management.",
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato Bacterial Spot",
        "description" : "Caused by Xanthomonas campestris. Small dark water-soaked spots on leaves and fruit with yellow halos.",
        "treatment"   : "Apply copper-based bactericides. Remove infected plant material immediately.",
        "prevention"  : "Use certified disease-free seeds. Avoid overhead irrigation. Rotate crops.",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "description" : "Caused by Alternaria solani. Dark brown spots with concentric rings on lower leaves first.",
        "treatment"   : "Apply fungicides containing chlorothalonil or mancozeb. Remove infected leaves.",
        "prevention"  : "Water at soil level. Space plants 45-60cm apart. Remove plant debris after harvest.",
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato Late Blight",
        "description" : "Caused by Phytophthora infestans. Large dark water-soaked lesions on leaves. Can destroy crop in days.",
        "treatment"   : "Immediately apply fungicides (cymoxanil + mancozeb). Remove and destroy infected plants.",
        "prevention"  : "Plant resistant varieties. Avoid overhead watering. Monitor weather forecasts.",
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato Leaf Mold",
        "description" : "Caused by Cladosporium fulvum. Yellow patches on upper leaf with olive-green mold on undersides.",
        "treatment"   : "Improve ventilation. Apply chlorothalonil or copper fungicides.",
        "prevention"  : "Maintain humidity below 85%. Space plants properly.",
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato Septoria Leaf Spot",
        "description" : "Caused by Septoria lycopersici. Small circular spots with dark borders on lower leaves.",
        "treatment"   : "Apply fungicides (chlorothalonil). Remove infected lower leaves.",
        "prevention"  : "Rotate crops. Remove plant debris. Water at base of plant.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato Spider Mites",
        "description" : "Tetranychus urticae infestation. Tiny mites cause stippled yellowing leaves. Fine webbing visible.",
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
        "description" : "Viral disease transmitted by whiteflies. Leaves curl upward and turn yellow. No cure.",
        "treatment"   : "No direct treatment. Remove infected plants immediately. Control whitefly populations.",
        "prevention"  : "Use whitefly-resistant varieties. Install insect-proof nets.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus",
        "description" : "Highly contagious RNA virus causing mottled light/dark green mosaic on leaves.",
        "treatment"   : "No cure. Remove infected plants. Disinfect all tools with 10% bleach solution.",
        "prevention"  : "Use certified virus-free seeds. Control aphids. Plant resistant varieties.",
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato Plant",
        "description" : "Your tomato plant appears healthy! Leaves show normal green coloration with no spots.",
        "treatment"   : "No treatment needed. Maintain regular watering and fertilization.",
        "prevention"  : "Continue current practices. Regularly inspect for early signs of disease.",
    },
    "default": {
        "display_name": "Plant Disease Detected",
        "description" : "A plant disease has been detected. Please consult local agricultural extension services.",
        "treatment"   : "Consult a local agricultural expert for proper identification and treatment plan.",
        "prevention"  : "Practice good crop rotation, proper irrigation, and regular plant inspection.",
    }
}

model     = None
class_labels = {}

def load_model():
    global model, class_labels
    model_path  = _resolve_path(app.config['MODEL_PATH'])
    labels_path = _resolve_path(app.config['LABELS_PATH'])

    if model_path.exists():
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            model = None
    else:
        logger.warning(f"Model not found at {model_path} - running in demo mode")

    if labels_path.exists():
        with open(labels_path) as f:
            raw = json.load(f)
        class_labels = {int(k): v for k, v in raw.items()}
        logger.info(f"Labels loaded: {len(class_labels)} classes")
    else:
        logger.warning("Labels file not found")

    if model is not None and class_labels:
        try:
            output_classes = int(model.output_shape[-1])
            if output_classes != len(class_labels):
                logger.warning(
                    f"Model output classes ({output_classes}) do not match labels ({len(class_labels)}). "
                    "Predictions may be inaccurate."
                )
        except Exception as e:
            logger.warning(f"Could not validate model/label consistency: {e}")

def preprocess_image(image_bytes):
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    resized  = pil_img.resize(app.config['IMG_SIZE'], Image.Resampling.BILINEAR)
    rgb_img  = np.asarray(resized, dtype=np.uint8)

    # Keep inference preprocessing aligned with training by default.
    if app.config.get("USE_CLAHE_PREPROCESS", False):
        lab      = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        l, a, b  = cv2.split(lab)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l        = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        rgb_img  = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    arr      = rgb_img.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def determine_severity(confidence):
    if confidence < 0.60:
        return "Mild"
    elif confidence < 0.85:
        return "Moderate"
    else:
        return "Severe"


def get_disease_info(class_name, confidence):
    info   = DISEASE_INFO.get(class_name, DISEASE_INFO["default"])
    result = {
        "disease_name": info["display_name"],
        "description" : info["description"],
        "treatment"   : info["treatment"],
        "prevention"  : info["prevention"],
        "severity"    : determine_severity(confidence),
        "is_healthy"  : "healthy" in class_name.lower(),
    }
    return result

def demo_prediction():
    import random
    demo_classes = [k for k in DISEASE_INFO.keys() if k != "default"]
    demo_class   = random.choice(demo_classes)
    confidence   = round(random.uniform(0.75, 0.98), 4)
    return demo_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.utcnow().isoformat(),
        "version"     : "1.0.0",
        "classes"     : len(class_labels)
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        "total"  : len(DISEASE_INFO),
        "classes": [info["display_name"] for key, info in DISEASE_INFO.items() if key != "default"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    logger.info(f"Prediction request received from {request.remote_addr}")

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    if ext not in app.config['ALLOWED_EXTS']:
        return jsonify({"error": f"Unsupported file type '.{ext}'."}), 400

    image_bytes = file.read()

    if len(image_bytes) > app.config['MAX_FILE_SIZE']:
        return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty file provided."}), 400

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 422

    try:
        if model is not None:
            predictions = model.predict(img_array, verbose=0)
            if predictions.ndim != 2 or predictions.shape[0] == 0:
                raise ValueError(f"Unexpected prediction shape: {predictions.shape}")

            pred_vector = predictions[0].astype(np.float32)
            score_sum = float(np.sum(pred_vector))

            # Handle models that output logits instead of softmax probabilities.
            if np.min(pred_vector) < 0.0 or score_sum < 0.99 or score_sum > 1.01:
                pred_vector = tf.nn.softmax(pred_vector).numpy()

            class_idx   = int(np.argmax(pred_vector))
            confidence  = float(pred_vector[class_idx])
            class_name  = class_labels.get(class_idx, "Unknown")
            top3_indices = np.argsort(pred_vector)[-3:][::-1]
            top3 = [
                {"class": class_labels.get(int(i), "Unknown"), "confidence": float(pred_vector[i])}
                for i in top3_indices
            ]
        else:
            logger.warning("Model not loaded - using demo mode")
            class_name, confidence = demo_prediction()
            top3 = [{"class": class_name, "confidence": confidence}]

        if confidence < app.config['CONFIDENCE_THRESHOLD']:
            return jsonify({
                "error"     : "Could not confidently identify disease. Please upload a clearer image.",
                "confidence": round(confidence, 4)
            }), 200

    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    lang         = request.form.get('lang', 'en')
    disease_info = get_disease_info(class_name, confidence)
    ai_info      = get_ai_description(
        disease_info["disease_name"],
        confidence,
        disease_info["is_healthy"],
        lang
    )

    if ai_info:
        ai_description = ai_info.get("description")
        ai_treatment   = ai_info.get("treatment")
        ai_prevention  = ai_info.get("prevention")

        disease_info["description_en"] = ai_description if lang == "en" and ai_description else disease_info["description"]
        disease_info["treatment_en"]   = ai_treatment if lang == "en" and ai_treatment else disease_info["treatment"]
        disease_info["prevention_en"]  = ai_prevention if lang == "en" and ai_prevention else disease_info["prevention"]
        disease_info["description_hi"] = ai_description if lang == "hi" and ai_description else "Hindi translation not available."
        disease_info["treatment_hi"]   = ai_treatment if lang == "hi" and ai_treatment else "Hindi translation not available."
        disease_info["prevention_hi"]  = ai_prevention if lang == "hi" and ai_prevention else "Hindi translation not available."
        disease_info["ai_powered"]  = True
    else:
        disease_info["description_en"] = disease_info["description"]
        disease_info["treatment_en"]   = disease_info["treatment"]
        disease_info["prevention_en"]  = disease_info["prevention"]
        disease_info["description_hi"] = "Hindi translation not available."
        disease_info["treatment_hi"]   = "Hindi translation not available."
        disease_info["prevention_hi"]  = "Hindi translation not available."
        disease_info["ai_powered"]  = False

    processing_time = round(time.time() - start_time, 3)

    response = {
        **disease_info,
        "confidence_score" : round(confidence, 4),
        "raw_class"        : class_name,
        "processing_time"  : processing_time,
        "top_predictions"  : top3 if model is not None else [],
    }

    logger.info(f"Prediction: {class_name} | Confidence: {confidence:.2%} | Time: {processing_time}s")
    return jsonify(response), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

with app.app_context():
    load_model()

if __name__ == '__main__':
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    logger.info(f"Starting Plant Disease Detection API on port {port}")
    logger.info(f"Visit: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
