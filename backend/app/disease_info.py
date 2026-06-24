"""
PlantVillage Disease Classes + Treatment Database
38 classes across 14 crops from the PlantVillage dataset
"""

# All 38 class names matching PlantVillage dataset labels
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Tomato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark spots with concentric rings (target-like) on lower leaves, spreading upward.",
        "treatment": [
            "Apply copper-based fungicides or chlorothalonil",
            "Remove and destroy infected leaves immediately",
            "Apply mancozeb or azoxystrobin fungicide",
            "Spray every 7–10 days during wet weather",
        ],
        "prevention": [
            "Use certified disease-free seeds",
            "Rotate crops — avoid tomatoes in the same spot for 3 years",
            "Mulch around plants to prevent soil splash",
            "Water at the base, avoid wetting foliage",
        ],
    },
    "Tomato___Late_blight": {
        "severity": "Severe",
        "description": "Highly destructive oomycete disease. Water-soaked lesions on leaves and fruit. Can destroy entire crop rapidly.",
        "treatment": [
            "Apply mancozeb + metalaxyl (Ridomil) immediately",
            "Remove and destroy all infected plant material",
            "Apply copper hydroxide as protective spray",
            "Increase ventilation between plants",
        ],
        "prevention": [
            "Plant resistant varieties (e.g., Mountain Magic, Defiant)",
            "Avoid overhead irrigation",
            "Space plants adequately for air circulation",
            "Apply preventive fungicide before wet periods",
        ],
    },
    "Tomato___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves and fruit.",
        "treatment": [
            "Apply copper-based bactericide sprays",
            "Use fixed copper + mancozeb combination",
            "Remove heavily infected plant parts",
            "Apply streptomycin in severe cases",
        ],
        "prevention": [
            "Use disease-free, certified seeds",
            "Avoid working with wet plants",
            "Disinfect tools between plants",
            "Rotate crops with non-solanaceous plants",
        ],
    },
    "Tomato___Leaf_Mold": {
        "severity": "Low-Moderate",
        "description": "Fungal disease thriving in high humidity. Olive-green mold on leaf undersides, yellow spots on upper surface.",
        "treatment": [
            "Apply fungicides containing chlorothalonil or mancozeb",
            "Improve greenhouse ventilation significantly",
            "Remove and dispose of infected leaves",
        ],
        "prevention": [
            "Maintain humidity below 85%",
            "Plant resistant varieties",
            "Ensure proper spacing for airflow",
            "Avoid overhead watering",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Fungal disease causing circular spots with dark borders and light centers on lower leaves.",
        "treatment": [
            "Apply copper-based or chlorothalonil fungicides",
            "Remove infected lower leaves",
            "Apply mancozeb every 7–10 days",
        ],
        "prevention": [
            "Mulch to reduce soil splash",
            "Avoid wetting foliage",
            "Practice crop rotation",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Tiny mites cause bronzing, stippling, and webbing on leaves. Thrive in hot, dry conditions.",
        "treatment": [
            "Apply miticides: abamectin, bifenazate, or spiromesifen",
            "Use insecticidal soap or neem oil spray",
            "Introduce predatory mites (Phytoseiulus persimilis)",
            "Increase irrigation to raise humidity",
        ],
        "prevention": [
            "Monitor plants regularly with magnifying lens",
            "Avoid dusty conditions",
            "Maintain adequate soil moisture",
        ],
    },
    "Tomato___Target_Spot": {
        "severity": "Moderate",
        "description": "Fungal spots with concentric rings on leaves and fruit. Favored by warm, wet conditions.",
        "treatment": [
            "Apply azoxystrobin or boscalid fungicide",
            "Remove infected plant material",
        ],
        "prevention": [
            "Crop rotation",
            "Proper plant spacing",
            "Avoid overhead irrigation",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Severe",
        "description": "Viral disease spread by whiteflies. Leaves curl upward and turn yellow. Severely reduces yield.",
        "treatment": [
            "No cure — remove and destroy infected plants",
            "Control whitefly vectors with imidacloprid or pyrethroids",
            "Use yellow sticky traps to monitor whitefly",
        ],
        "prevention": [
            "Plant resistant varieties (e.g., Ty-1 gene varieties)",
            "Use insect-proof netting in greenhouse",
            "Apply reflective mulch to deter whiteflies",
            "Remove volunteer tomato plants",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "severity": "Moderate-Severe",
        "description": "Viral disease causing mosaic patterns, leaf distortion, and reduced fruit quality.",
        "treatment": [
            "No chemical cure — remove infected plants",
            "Wash hands thoroughly when handling plants",
        ],
        "prevention": [
            "Use virus-free, certified seeds",
            "Disinfect tools with 10% bleach solution",
            "Control aphid vectors",
            "Avoid smoking near plants (tobacco mosaic virus spread)",
        ],
    },
    "Tomato___healthy": {
        "severity": "None",
        "description": "Plant appears healthy with no visible disease symptoms.",
        "treatment": [],
        "prevention": [
            "Continue regular monitoring",
            "Maintain proper irrigation and fertilization",
            "Practice crop rotation annually",
        ],
    },
    "Potato___Early_blight": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark, target-like spots on older lower leaves. Reduces yield.",
        "treatment": [
            "Apply mancozeb or chlorothalonil fungicide",
            "Remove infected foliage",
            "Apply every 7–14 days during wet conditions",
        ],
        "prevention": [
            "Use certified seed potatoes",
            "Rotate crops for 3+ years",
            "Avoid overhead irrigation",
        ],
    },
    "Potato___Late_blight": {
        "severity": "Severe",
        "description": "The disease behind the Irish Famine. Water-soaked lesions rapidly destroy foliage and tubers.",
        "treatment": [
            "Apply metalaxyl + mancozeb (Ridomil Gold MZ) immediately",
            "Use fluazinam or cymoxanil fungicide",
            "Remove and destroy all infected material",
        ],
        "prevention": [
            "Plant resistant varieties",
            "Apply prophylactic fungicide before disease onset",
            "Destroy volunteer plants and cull piles",
        ],
    },
    "Apple___Apple_scab": {
        "severity": "Moderate",
        "description": "Fungal disease causing olive-brown scab lesions on leaves and fruit. Reduces marketability.",
        "treatment": [
            "Apply myclobutanil, captan, or mancozeb fungicide",
            "Begin spraying at green tip stage",
        ],
        "prevention": [
            "Plant scab-resistant varieties",
            "Rake and destroy fallen leaves",
            "Prune for good air circulation",
        ],
    },
    "Apple___Black_rot": {
        "severity": "Moderate-Severe",
        "description": "Fungal disease causing frogeye leaf spot and mummified fruit. Overwinters in dead wood.",
        "treatment": [
            "Apply captan or thiophanate-methyl",
            "Remove mummified fruit and cankers",
        ],
        "prevention": [
            "Prune out dead wood in dormant season",
            "Apply dormant copper spray",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "severity": "Moderate",
        "description": "Fungal disease requiring both apple and cedar/juniper hosts. Causes orange spots on leaves.",
        "treatment": [
            "Apply myclobutanil or propiconazole during infection periods",
        ],
        "prevention": [
            "Remove nearby juniper/cedar trees if possible",
            "Plant resistant apple varieties",
        ],
    },
    "Apple___healthy": {
        "severity": "None",
        "description": "Healthy apple leaf with no disease symptoms.",
        "treatment": [],
        "prevention": ["Regular monitoring", "Annual pruning", "Proper nutrition"],
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Moderate",
        "description": "Fungal disease causing powdery orange/brown pustules on both leaf surfaces.",
        "treatment": [
            "Apply triazole fungicides (propiconazole, tebuconazole)",
            "Spray early when first pustules appear",
        ],
        "prevention": [
            "Plant resistant hybrids",
            "Avoid late planting dates",
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "severity": "Moderate-Severe",
        "description": "Fungal disease causing long, cigar-shaped gray-green lesions. Major yield reducer.",
        "treatment": [
            "Apply azoxystrobin or propiconazole at VT/R1 stage",
        ],
        "prevention": [
            "Use resistant hybrids",
            "Crop rotation with non-host crops",
            "Reduce residue with tillage",
        ],
    },
    "Grape___Black_rot": {
        "severity": "Severe",
        "description": "Fungal disease causing circular lesions on leaves and mummified berries. Major yield loss.",
        "treatment": [
            "Apply mancozeb, captan, or myclobutanil",
            "Remove all mummified berries",
            "Begin spraying at bud break",
        ],
        "prevention": [
            "Prune for open canopy and airflow",
            "Remove and destroy mummified fruit",
        ],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "severity": "Lethal",
        "description": "Bacterial disease spread by Asian citrus psyllid. No cure. Eventually kills trees.",
        "treatment": [
            "No effective cure exists",
            "Remove and destroy infected trees to prevent spread",
            "Control psyllid vector aggressively with insecticides",
        ],
        "prevention": [
            "Use certified disease-free planting material",
            "Monitor for and control Asian citrus psyllid",
            "Report suspected cases to agricultural authorities",
        ],
    },
    "Squash___Powdery_mildew": {
        "severity": "Low-Moderate",
        "description": "Fungal disease causing white powdery coating on leaves. Reduces photosynthesis.",
        "treatment": [
            "Apply sulfur-based or potassium bicarbonate fungicide",
            "Neem oil spray is effective",
            "Apply baking soda solution (1 tbsp per liter water)",
        ],
        "prevention": [
            "Maintain good air circulation",
            "Avoid overhead watering",
            "Plant resistant varieties",
        ],
    },
    "Strawberry___Leaf_scorch": {
        "severity": "Moderate",
        "description": "Fungal disease causing dark purple spots that enlarge and cause leaf scorching.",
        "treatment": [
            "Apply captan or myclobutanil fungicide",
            "Remove infected leaves",
        ],
        "prevention": [
            "Plant resistant cultivars",
            "Avoid overhead irrigation",
            "Renovate beds after harvest",
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Bacterial disease causing water-soaked spots on leaves and fruit with yellow halos.",
        "treatment": [
            "Copper-based bactericides",
            "Remove infected plant material",
        ],
        "prevention": [
            "Use certified disease-free seed",
            "Avoid working with wet plants",
            "Crop rotation",
        ],
    },
}

# Add healthy entries for all unlisted healthy classes
for cls in CLASS_NAMES:
    if "healthy" in cls and cls not in DISEASE_INFO:
        DISEASE_INFO[cls] = {
            "severity": "None",
            "description": "Plant appears healthy with no visible disease symptoms.",
            "treatment": [],
            "prevention": ["Continue regular monitoring", "Maintain proper care"],
        }

# PlantDoc detection labels use different names than the PlantVillage classifier.
# These aliases keep the old disease descriptions and cure tips available when
# the bounding-box model returns PlantDoc class names.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Grape leaf": "Grape___healthy",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}

GENERIC_HEALTHY_LEAF_INFO = {
    "severity": "None",
    "description": "Plant leaf appears healthy with no visible disease symptoms.",
    "treatment": [],
    "prevention": ["Continue regular monitoring", "Maintain proper irrigation and nutrition"],
}

GENERIC_DISEASE_INFO = {
    "severity": "Unknown",
    "description": "Disease detected by the bounding-box model. Specific treatment notes are not available for this class yet.",
    "treatment": ["Isolate affected leaves if possible", "Consult a local agronomist before applying chemicals"],
    "prevention": ["Avoid overhead watering", "Improve airflow around plants", "Remove heavily infected leaves"],
}


def get_disease_info(class_name: str) -> dict:
    """Return treatment metadata for classifier or PlantDoc detection labels."""
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    alias = PLANTDOC_TO_PLANTVILLAGE.get(class_name)
    if alias and alias in DISEASE_INFO:
        return DISEASE_INFO[alias]

    if "leaf" in class_name.lower() and not any(word in class_name.lower() for word in ["spot", "rust", "blight", "mildew", "virus", "mold", "scab", "rot", "mites"]):
        return GENERIC_HEALTHY_LEAF_INFO

    return GENERIC_DISEASE_INFO
