from pathlib import Path
from io import BytesIO
import colorsys
import tempfile
import shutil
import base64
import uuid
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))
from train_matcher import MatchTrainer

MODEL_NAME = "openai/clip-vit-base-patch32"
ALPHA = 0.4  # Image similarity weight
BETA = 0.4   # Metadata score weight (increased for better color matching!)
GAMMA = 0.2  # Training boost weight

app = FastAPI(title="Saree Blouse Matcher API")

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for reference saree images
reference_sarees = {}

COLOR_MATCH_SCORES = {
    # Traditional excellent matches
    ("red", "gold"): 0.95,
    ("maroon", "gold"): 0.95,
    ("green", "gold"): 0.90,
    ("blue", "silver"): 0.90,
    ("pink", "green"): 0.85,
    ("black", "red"): 0.80,
    ("black", "gold"): 0.90,
    ("white", "red"): 0.80,
    ("white", "gold"): 0.90,
    ("purple", "gold"): 0.85,
    ("purple", "purple"): 1.0,
    ("green", "green"): 1.0,
    ("red", "red"): 1.0,
    ("blue", "blue"): 1.0,
    ("pink", "pink"): 1.0,
    ("maroon", "maroon"): 1.0,
    ("black", "black"): 1.0,
    
    # Purple saree detection fix - it detects as black/brown!
    ("black", "purple"): 0.90,  # Dark purple looks black
    ("brown", "purple"): 0.90,  # Purple-brown shade
    ("gray", "purple"): 0.75,   # Grayish purple
    
    # Silver/Gray matches - IMPORTANT!
    ("silver", "silver"): 1.0,
    ("gray", "silver"): 0.95,
    ("silver", "gray"): 0.95,
    ("gray", "gray"): 1.0,
    ("black", "silver"): 0.90,
    ("white", "silver"): 0.85,
    
    # Gray + Gold = BAD MATCH!
    ("gray", "gold"): 0.30,
    ("gold", "gray"): 0.30,
    ("silver", "gold"): 0.25,
    ("gold", "silver"): 0.25,
    
    # Purple mismatches
    ("purple", "green"): 0.15,
    ("purple", "silver"): 0.20,
    
    # Gold specific - NOT neutral with everything!
    ("gold", "gold"): 1.0,
    ("brown", "gold"): 0.70,
    ("yellow", "gold"): 0.85,
    ("black", "gold"): 0.40,  # Dark colors don't match gold well
    
    # Green mismatches
    ("green", "purple"): 0.15,
    ("green", "silver"): 0.20,
    ("green", "black"): 0.30,
}

COLOR_PROTOTYPES = {
    "red": np.array([200, 50, 60]),
    "maroon": np.array([128, 0, 0]),
    "pink": np.array([230, 140, 180]),
    "green": np.array([50, 140, 70]),
    "blue": np.array([60, 90, 200]),
    "purple": np.array([130, 70, 170]),
    "yellow": np.array([220, 210, 60]),
    "gold": np.array([212, 175, 55]),
    "silver": np.array([180, 180, 190]),
    "white": np.array([245, 245, 245]),
    "black": np.array([20, 20, 20]),
    "beige": np.array([220, 200, 170]),
    "cream": np.array([245, 235, 210]),
    "orange": np.array([230, 130, 40]),
    "brown": np.array([120, 80, 50]),
    "gray": np.array([128, 128, 128]),
}

model = None
processor = None
catalog_df = None
embeddings = None
trainer = None  # Training system for learned matches
yolo_model = None  # YOLO for object detection
device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms

def load_image(image_path: Path):
    return Image.open(image_path).convert("RGB")

def load_image_from_bytes(data: bytes):
    return Image.open(BytesIO(data)).convert("RGB")

def closest_color_name(rgb: np.ndarray) -> str:
    best_name = None
    best_distance = None
    for name, proto in COLOR_PROTOTYPES.items():
        dist = np.linalg.norm(rgb - proto)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_name = name
    return best_name

def rgb_to_hsv_features(rgb_pixels: np.ndarray):
    hsv_values = []
    for r, g, b in rgb_pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        hsv_values.append((h, s, v))
    return np.array(hsv_values)

def filter_background_pixels(pixels: np.ndarray):
    hsv = rgb_to_hsv_features(pixels)
    value = hsv[:, 2]
    saturation = hsv[:, 1]

    not_too_white = value < 0.95
    not_too_black = value > 0.12
    not_too_gray = saturation > 0.15

    mask = not_too_white & not_too_black & not_too_gray
    filtered = pixels[mask]

    if len(filtered) < 200:
        relaxed_mask = not_too_white & not_too_black
        filtered = pixels[relaxed_mask]

    if len(filtered) < 100:
        filtered = pixels

    return filtered

def detect_dominant_color_with_clip(image: Image.Image) -> str:
    """
    Use CLIP to detect the dominant color by comparing with text descriptions.
    Much more accurate than RGB clustering!
    """
    # Color descriptions to test
    color_descriptions = [
        "a purple colored fabric",
        "a gold colored fabric", 
        "a silver colored fabric",
        "a green colored fabric",
        "a red colored fabric",
        "a blue colored fabric",
        "a black colored fabric",
        "a white colored fabric",
        "a pink colored fabric",
        "a brown colored fabric",
        "a gray colored fabric",
        "a yellow colored fabric",
        "an orange colored fabric",
    ]
    
    # Get image features (use vision_model + projection)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.inference_mode():
        vision_output = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_output.pooler_output)
    
    # Get text features (use text_model + projection)
    text_inputs = processor(text=color_descriptions, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    
    with torch.inference_mode():
        text_output = model.text_model(**text_inputs)
        text_features = model.text_projection(text_output.pooler_output)
    
    # Normalize and compute similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).cpu().numpy()[0]
    
    # Get best match and top 3
    top_indices = np.argsort(similarity)[::-1][:3]
    best_color = color_descriptions[top_indices[0]].split()[1]  # Extract color name
    confidence = float(similarity[top_indices[0]])
    
    print(f"  🎨 CLIP detected color: {best_color} (confidence: {confidence:.3f})")
    print(f"     Top 3: {[color_descriptions[i].split()[1] + f'({similarity[i]:.2f})' for i in top_indices]}")
    
    return best_color

def extract_color_palette_from_pil(image: Image.Image, n_clusters: int = 5, resize_to=(180, 180)):
    image = image.convert("RGB").resize(resize_to)
    pixels = np.array(image).reshape(-1, 3)

    filtered_pixels = filter_background_pixels(pixels)

    n_clusters = min(n_clusters, max(1, len(filtered_pixels)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(filtered_pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]
    total = counts.sum()

    palette = []
    for idx in order:
        rgb = centers[idx].astype(int)
        share = float(counts[idx] / total)
        palette.append({
            "name": closest_color_name(rgb),
            "rgb": rgb.tolist(),
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "share": round(share, 4),
        })

    return {
        "palette": palette,
        "total_pixels": int(len(pixels)),
        "filtered_pixels": int(len(filtered_pixels)),
    }

def embed_pil_image(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.vision_model(**inputs)
        features = outputs.pooler_output  # Get the pooled features tensor

    emb = features.cpu().numpy().astype("float32")
    return l2_normalize(emb)

def exact_match(a, b):
    return 1.0 if normalize_text(a) == normalize_text(b) and normalize_text(a) != "" else 0.0

def partial_match(a, b):
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.7
    return 0.0

def pair_color_score(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if (a, b) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(a, b)]
    if (b, a) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(b, a)]

    # Neutral colors get REDUCED score - not high!
    neutral_colors = {"black", "white", "beige", "cream"}
    if b in neutral_colors:
        return 0.40  # Reduced from 0.75!

    # No match at all
    return 0.10  # Reduced from 0.25 - be strict!

def palette_color_score(query_palette, blouse_color):
    blouse_color = normalize_text(blouse_color)
    if not blouse_color or not query_palette:
        return 0.0

    # Debug: Print palette colors
    palette_colors = [p["name"] for p in query_palette]
    print(f"  Saree palette colors: {palette_colors}")
    print(f"  Matching against blouse color: {blouse_color}")

    score = float(sum(pair_color_score(p["name"], blouse_color) * p["share"] for p in query_palette))
    print(f"  Palette color score: {score:.3f}")
    return score

def clip_color_score(saree_clip_color, blouse_color):
    """
    Direct CLIP color matching - much simpler and more accurate!
    """
    saree_clip_color = normalize_text(saree_clip_color)
    blouse_color = normalize_text(blouse_color)
    
    if not saree_clip_color or not blouse_color:
        return 0.0
    
    # Check explicit matches in COLOR_MATCH_SCORES
    if saree_clip_color == blouse_color:
        return 1.0
    if (saree_clip_color, blouse_color) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(saree_clip_color, blouse_color)]
    if (blouse_color, saree_clip_color) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(blouse_color, saree_clip_color)]
    
    # Default to low score if no match
    return 0.10

def fabric_score(query_fabric, blouse_fabric):
    query_fabric = normalize_text(query_fabric)
    blouse_fabric = normalize_text(blouse_fabric)

    if not query_fabric or not blouse_fabric:
        return 0.0
    if query_fabric == blouse_fabric:
        return 1.0

    compatible = {
        ("silk", "brocade"),
        ("silk", "silk"),
        ("cotton", "cotton"),
        ("georgette", "satin"),
        ("georgette", "silk"),
        ("chiffon", "satin"),
        ("net", "satin"),
    }

    if (query_fabric, blouse_fabric) in compatible or (blouse_fabric, query_fabric) in compatible:
        return 0.8

    return 0.3

def pattern_score(query_pattern, blouse_pattern):
    query_pattern = normalize_text(query_pattern)
    blouse_pattern = normalize_text(blouse_pattern)

    if not query_pattern or not blouse_pattern:
        return 0.0
    if query_pattern == blouse_pattern:
        return 0.9
    if query_pattern in {"zari", "embroidered"} and blouse_pattern in {"plain", "brocade"}:
        return 0.85
    if query_pattern in {"printed"} and blouse_pattern in {"plain"}:
        return 0.9
    if query_pattern in {"plain"} and blouse_pattern in {"embroidered", "zari"}:
        return 0.85
    return 0.35

def occasion_score(query_occasion, blouse_occasion):
    return max(exact_match(query_occasion, blouse_occasion), partial_match(query_occasion, blouse_occasion))

def style_score(query_style, blouse_style):
    return max(exact_match(query_style, blouse_style), partial_match(query_style, blouse_style))

def attribute_score(query_meta: dict, blouse_row):
    # Use CLIP color if available (more accurate!)
    if "clip_color" in query_meta and query_meta["clip_color"]:
        c = clip_color_score(query_meta["clip_color"], blouse_row.get("color"))
        print(f"  Using CLIP color: {query_meta['clip_color']} → {blouse_row.get('color')}: {c:.3f}")
    else:
        # Fallback to palette-based color scoring
        c = palette_color_score(query_meta.get("palette", []), blouse_row.get("color"))
    
    f = fabric_score(query_meta.get("fabric"), blouse_row.get("fabric"))
    p = pattern_score(query_meta.get("pattern"), blouse_row.get("pattern"))
    o = occasion_score(query_meta.get("occasion"), blouse_row.get("occasion"))
    s = style_score(query_meta.get("style"), blouse_row.get("style"))

    # Check if we have full metadata or just color
    has_fabric = bool(query_meta.get("fabric"))
    has_pattern = bool(query_meta.get("pattern"))
    has_occasion = bool(query_meta.get("occasion"))
    has_style = bool(query_meta.get("style"))
    
    # If we only have color (no other metadata), give color 100% weight
    if not (has_fabric or has_pattern or has_occasion or has_style):
        final_attr = c  # Color gets full weight!
    else:
        # Full metadata available, use balanced weights
        final_attr = 0.35 * c + 0.20 * f + 0.15 * p + 0.15 * o + 0.15 * s
    
    return final_attr, {
        "palette_color_score": c,
        "fabric_score": f,
        "pattern_score": p,
        "occasion_score": o,
        "style_score": s,
    }

def recommend_from_query(query_embedding, blouse_df, embeddings, query_meta, top_k=5, candidate_k=20):
    blouse_embedding_indices = blouse_df["embedding_index"].astype(int).to_numpy()
    blouse_vectors = embeddings[blouse_embedding_indices]

    sims = cosine_similarity(query_embedding, blouse_vectors)[0]
    result_df = blouse_df.copy()
    result_df["image_score"] = sims
    result_df = result_df.sort_values("image_score", ascending=False).head(candidate_k).copy()

    attr_scores = []
    breakdowns = []

    for _, row in result_df.iterrows():
        score, breakdown = attribute_score(query_meta, row)
        attr_scores.append(score)
        breakdowns.append(breakdown)

    result_df["attr_score"] = attr_scores
    result_df["palette_color_score"] = [b["palette_color_score"] for b in breakdowns]
    result_df["fabric_score"] = [b["fabric_score"] for b in breakdowns]
    result_df["pattern_score"] = [b["pattern_score"] for b in breakdowns]
    result_df["occasion_score"] = [b["occasion_score"] for b in breakdowns]
    result_df["style_score"] = [b["style_score"] for b in breakdowns]
    
    # Add training boost if available
    training_boosts = []
    if trainer and len(trainer.training_pairs) > 0:
        for idx, row in result_df.iterrows():
            blouse_idx = int(row["embedding_index"])
            blouse_emb = embeddings[blouse_idx]
            boost = trainer.get_learned_boost(
                query_embedding, 
                blouse_emb,
                query_meta,
                row.to_dict()
            )
            training_boosts.append(boost)
    else:
        training_boosts = [0.0] * len(result_df)
    
    result_df["training_boost"] = training_boosts
    result_df["final_score"] = (ALPHA * result_df["image_score"] + 
                                BETA * result_df["attr_score"] + 
                                GAMMA * result_df["training_boost"])

    result_df = result_df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
    return result_df

@app.on_event("startup")
def startup_event():
    global model, processor, catalog_df, embeddings, trainer, yolo_model

    project_root = Path(__file__).resolve().parent
    catalog_path = project_root / "outputs" / "catalog_with_status.csv"
    embeddings_path = project_root / "outputs" / "embeddings.npy"

    if not catalog_path.exists():
        raise RuntimeError(f"Missing file: {catalog_path}")
    if not embeddings_path.exists():
        raise RuntimeError(f"Missing file: {embeddings_path}")

    catalog_df = pd.read_csv(catalog_path)
    embeddings = np.load(embeddings_path)

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Initialize training system
    trainer = MatchTrainer()
    print(f"Loaded {len(trainer.training_pairs)} training pairs")
    
    # Initialize YOLO model
    try:
        # Try to load custom model first, fallback to pretrained
        custom_model_path = project_root / "models" / "blouse_detector.pt"
        if custom_model_path.exists():
            yolo_model = YOLO(str(custom_model_path))
            print(f"✓ Loaded custom YOLO model from {custom_model_path}")
        else:
            # Use pretrained YOLOv8 nano model
            yolo_model = YOLO('yolov8n.pt')
            print("✓ Loaded pretrained YOLOv8n model (will detect 'person' class as proxy)")
            print("  Note: For better results, train on fashion dataset")
    except Exception as e:
        print(f"⚠ Warning: Could not load YOLO model: {e}")
        print("  YOLO detection will be unavailable")
        yolo_model = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_NAME,
        "catalog_rows": int(len(catalog_df)) if catalog_df is not None else 0,
        "yolo_available": yolo_model is not None,
        "detection_methods": ["grid", "yolo"] if yolo_model else ["grid"]
    }

@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    candidate_k: int = Form(20),
    fabric: str = Form(""),
    pattern: str = Form(""),
    occasion: str = Form(""),
    style: str = Form("")
):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)

        query_embedding = embed_pil_image(image)
        palette_info = extract_color_palette_from_pil(image)

        blouse_df = catalog_df[catalog_df["item_type"].str.lower() == "blouse"].copy()
        if blouse_df.empty:
            raise HTTPException(status_code=500, detail="No blouse items found in catalog")

        query_meta = {
            "palette": palette_info["palette"],
            "fabric": fabric,
            "pattern": pattern,
            "occasion": occasion,
            "style": style,
        }

        recommendations = recommend_from_query(
            query_embedding=query_embedding,
            blouse_df=blouse_df,
            embeddings=embeddings,
            query_meta=query_meta,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        result_columns = [
            "item_id", "item_type", "image_path", "color", "fabric", "pattern",
            "occasion", "style", "image_score", "palette_color_score",
            "attr_score", "final_score"
        ]
        result_columns = [c for c in result_columns if c in recommendations.columns]

        return {
            "filename": file.filename,
            "detected_palette": palette_info["palette"],
            "total_pixels": palette_info["total_pixels"],
            "filtered_pixels": palette_info["filtered_pixels"],
            "recommendations": recommendations[result_columns].to_dict(orient="records")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store_saree")
async def store_saree(
    file: UploadFile = File(...),
    saree_id: str = Form(None)
):
    """
    Store a reference saree image for later matching.
    Returns a saree_id that can be used for detection.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        
        # Generate ID if not provided
        if not saree_id:
            saree_id = str(uuid.uuid4())
        
        print(f"📸 Processing saree: {file.filename}")
        # Use CLIP to detect dominant color (ML-based!)
        dominant_color = detect_dominant_color_with_clip(image)
        print(f"✓ CLIP color detection complete: {dominant_color}")
        
        # Extract embedding
        saree_embedding = embed_pil_image(image)
        palette_info = extract_color_palette_from_pil(image)
        
        # Add CLIP-detected color to palette info
        palette_info["clip_color"] = dominant_color
        
        # Store in memory
        reference_sarees[saree_id] = {
            "embedding": saree_embedding,
            "palette": palette_info["palette"],
            "clip_color": dominant_color,  # Store CLIP color!
            "filename": file.filename,
            "image_bytes": contents
        }
        
        return {
            "saree_id": saree_id,
            "message": "Saree stored successfully",
            "filename": file.filename,
            "palette": palette_info["palette"],
            "clip_detected_color": dominant_color  # Show CLIP color to user!
        }
    
    except Exception as e:
        import traceback
        print(f"❌ ERROR in store_saree: {str(e)}")
        traceback.print_exc()  # Print full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_blouse")
async def detect_blouse(
    file: UploadFile = File(...),
    saree_id: str = Form(...),
    grid_cols: int = Form(3),
    grid_rows: int = Form(2),
    draw_box: bool = Form(True)
):
    """
    Detect matching blouse in a composite image or scene with multiple blouses.
    Draws bounding box around the best match.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    if saree_id not in reference_sarees:
        raise HTTPException(status_code=404, detail=f"Saree ID '{saree_id}' not found. Please store saree first.")
    
    try:
        contents = await file.read()
        scene_image = load_image_from_bytes(contents)
        
        # Get reference saree data
        ref_data = reference_sarees[saree_id]
        query_embedding = ref_data["embedding"]
        query_palette = ref_data["palette"]
        
        # Get blouse catalog
        blouse_df = catalog_df[catalog_df["item_type"].str.lower() == "blouse"].copy()
        if blouse_df.empty:
            raise HTTPException(status_code=500, detail="No blouse items found in catalog")
        
        # Divide the scene image into grid cells
        img_width, img_height = scene_image.size
        cell_width = img_width // grid_cols
        cell_height = img_height // grid_rows
        
        best_score = -1
        best_position = None
        best_item = None
        cell_results = []
        
        # Scan each grid cell
        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Extract cell image
                cell_img = scene_image.crop((x1, y1, x2, y2))
                
                # Get embedding for this cell
                cell_embedding = embed_pil_image(cell_img)
                
                # Calculate similarity with all blouses
                blouse_indices = blouse_df["embedding_index"].astype(int).to_numpy()
                blouse_embeddings = embeddings[blouse_indices]
                
                sims = cosine_similarity(cell_embedding, blouse_embeddings)[0]
                best_idx = np.argmax(sims)
                max_sim = sims[best_idx]
                
                # Get metadata scores
                blouse_row = blouse_df.iloc[best_idx]
                blouse_emb = embeddings[int(blouse_row["embedding_index"])]
                
                # Use CLIP color for matching (Grid detection)!
                query_meta = {
                    "clip_color": ref_data.get("clip_color"),
                    "palette": query_palette,
                    "fabric": "",
                    "pattern": "",
                    "occasion": "",
                    "style": ""
                }
                
                attr, breakdown = attribute_score(query_meta, blouse_row)
                
                # Get training boost
                training_boost = 0.0
                if trainer and len(trainer.training_pairs) > 0:
                    training_boost = trainer.get_learned_boost(
                        query_embedding,
                        blouse_emb,
                        {"palette": query_palette},
                        blouse_row.to_dict()
                    )
                
                final = ALPHA * max_sim + BETA * attr + GAMMA * training_boost
                
                cell_results.append({
                    "row": row,
                    "col": col,
                    "bbox": [x1, y1, x2, y2],
                    "image_score": float(max_sim),
                    "attr_score": float(attr),
                    "training_boost": float(training_boost),
                    "final_score": float(final),
                    "item_id": blouse_row["item_id"],
                    "color": blouse_row["color"]
                })
                
                if final > best_score:
                    best_score = final
                    best_position = (x1, y1, x2, y2)
                    best_item = blouse_row
        
        # Draw bounding box if requested
        result_image = scene_image.copy()
        if draw_box and best_position:
            draw = ImageDraw.Draw(result_image)
            x1, y1, x2, y2 = best_position
            
            # Draw thick rectangle
            box_color = (0, 255, 0)  # Green
            thickness = 5
            for i in range(thickness):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=box_color, width=2)
            
            # Add label
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            label = f"Match: {best_item['item_id']} ({best_score:.2f})"
            text_bbox = draw.textbbox((x1, y1-30), label, font=font)
            draw.rectangle(text_bbox, fill=box_color)
            draw.text((x1, y1-30), label, fill=(0, 0, 0), font=font)
        
        # Convert result to bytes
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Return annotated image
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Best-Item-ID": str(best_item["item_id"]),
                "X-Best-Score": str(best_score),
                "X-Best-Bbox": f"{best_position[0]},{best_position[1]},{best_position[2]},{best_position[3]}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_blouse_yolo")
async def detect_blouse_yolo(
    file: UploadFile = File(...),
    saree_id: str = Form(...),
    confidence: float = Form(0.05),  # Much lower to detect all objects!
    draw_box: bool = Form(True),
    return_json: bool = Form(False)  # NEW: Return JSON for live detection
):
    """
    Detect matching blouse using YOLO object detection + CLIP matching.
    This is the REAL object detection approach (better than grid-based).
    """
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not available. Using grid detection instead.")
    
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    if saree_id not in reference_sarees:
        raise HTTPException(status_code=404, detail=f"Saree ID '{saree_id}' not found. Please store saree first.")
    
    try:
        contents = await file.read()
        scene_image = load_image_from_bytes(contents)
        scene_array = np.array(scene_image)
        
        # Get reference saree data
        ref_data = reference_sarees[saree_id]
        query_embedding = ref_data["embedding"]
        query_palette = ref_data["palette"]
        
        # Get blouse catalog
        blouse_df = catalog_df[catalog_df["item_type"].str.lower() == "blouse"].copy()
        if blouse_df.empty:
            raise HTTPException(status_code=500, detail="No blouse items found in catalog")
        
        # Run YOLO detection
        results = yolo_model(scene_array, conf=confidence, verbose=False)
        
        # Extract detected objects
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Crop detected region
                detected_region = scene_image.crop((x1, y1, x2, y2))
                
                # Extract CLIP embedding for this region
                region_embedding = embed_pil_image(detected_region)
                
                # Compare with all blouses in catalog
                blouse_indices = blouse_df["embedding_index"].astype(int).to_numpy()
                blouse_embeddings = embeddings[blouse_indices]
                
                sims = cosine_similarity(region_embedding, blouse_embeddings)[0]
                best_idx = np.argmax(sims)
                max_sim = sims[best_idx]
                
                # Get metadata scores
                blouse_row = blouse_df.iloc[best_idx]
                blouse_emb = embeddings[int(blouse_row["embedding_index"])]
                
                # Use CLIP color for matching (YOLO detection)!
                query_meta = {
                    "clip_color": ref_data.get("clip_color"),
                    "palette": query_palette,
                    "fabric": "",
                    "pattern": "",
                    "occasion": "",
                    "style": ""
                }
                
                attr, breakdown = attribute_score(query_meta, blouse_row)
                
                # Get training boost
                training_boost = 0.0
                if trainer and len(trainer.training_pairs) > 0:
                    training_boost = trainer.get_learned_boost(
                        query_embedding,
                        blouse_emb,
                        {"palette": query_palette},
                        blouse_row.to_dict()
                    )
                
                final_score = ALPHA * max_sim + BETA * attr + GAMMA * training_boost
                
                # Debug: Print scores for each detection
                print(f"Detection {i+1}: {blouse_row['item_id']} ({blouse_row['color']})")
                print(f"  Visual similarity: {max_sim:.3f}")
                print(f"  Color match: {breakdown['palette_color_score']:.3f}")
                print(f"  Attr score: {attr:.3f}")
                print(f"  Final score: {final_score:.3f}")
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "yolo_confidence": conf,
                    "yolo_class": cls,
                    "image_score": float(max_sim),
                    "attr_score": float(attr),
                    "training_boost": float(training_boost),
                    "final_score": float(final_score),
                    "matched_item_id": blouse_row["item_id"],
                    "matched_color": blouse_row["color"]
                })
        
        if not detections:
            # No detections, return original image with message
            img_byte_arr = BytesIO()
            scene_image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            return StreamingResponse(
                img_byte_arr,
                media_type="image/jpeg",
                headers={
                    "X-Detection-Count": "0",
                    "X-Message": "No objects detected. Try adjusting confidence threshold."
                }
            )
        
        # Find best match
        best_detection = max(detections, key=lambda x: x["final_score"])
        
        # If return_json is True, return JSON response with all detections
        if return_json:
            return JSONResponse(content={
                "detections": [
                    {
                        "item_id": det["matched_item_id"],
                        "color": det["matched_color"],
                        "bbox": det["bbox"],  # [x1, y1, x2, y2]
                        "final_score": det["final_score"],
                        "yolo_confidence": det["yolo_confidence"]
                    }
                    for det in sorted(detections, key=lambda x: x["final_score"], reverse=True)
                ],
                "best_match": {
                    "item_id": best_detection["matched_item_id"],
                    "score": best_detection["final_score"],
                    "bbox": best_detection["bbox"]
                }
            })
        
        # Draw bounding boxes if requested
        result_image = scene_image.copy()
        if draw_box:
            draw = ImageDraw.Draw(result_image)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw all detections (light boxes)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                if det != best_detection:
                    # Draw gray box for other detections
                    for i in range(2):
                        draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=(128, 128, 128), width=1)
            
            # Draw best match (green box)
            x1, y1, x2, y2 = best_detection["bbox"]
            box_color = (0, 255, 0)  # Green
            thickness = 5
            
            for i in range(thickness):
                draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=box_color, width=2)
            
            # Add label
            label = f"Match: {best_detection['matched_item_id']} ({best_detection['final_score']:.2f})"
            
            # Draw background for text
            text_bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(text_bbox, fill=box_color)
            draw.text((x1, y1-25), label, fill=(0, 0, 0), font=font)
            
            # Add detection count
            info_text = f"Detected: {len(detections)} object(s)"
            draw.text((10, 10), info_text, fill=box_color, font=font_small)
        
        # Convert result to bytes
        img_byte_arr = BytesIO()
        result_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        # Return annotated image
        return StreamingResponse(
            img_byte_arr,
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(detections)),
                "X-Best-Item-ID": str(best_detection["matched_item_id"]),
                "X-Best-Score": str(best_detection["final_score"]),
                "X-Best-Bbox": f"{best_detection['bbox'][0]},{best_detection['bbox'][1]},{best_detection['bbox'][2]},{best_detection['bbox'][3]}",
                "X-YOLO-Confidence": str(best_detection["yolo_confidence"]),
                "X-Detection-Method": "YOLO"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_sarees")
def list_sarees():
    """List all stored reference sarees."""
    return {
        "sarees": [
            {
                "saree_id": sid,
                "filename": data["filename"],
                "palette": data["palette"]
            }
            for sid, data in reference_sarees.items()
        ]
    }


@app.post("/confirm_match")
async def confirm_match(
    saree_id: str = Form(...),
    blouse_id: str = Form(...),
    is_good_match: bool = Form(True)
):
    """
    Provide feedback on a saree-blouse match to improve future recommendations.
    
    Args:
        saree_id: ID of saree item or stored reference
        blouse_id: ID of matched blouse from catalog
        is_good_match: True if match is good, False if incorrect
    """
    try:
        # Get saree embedding
        saree_emb = None
        saree_meta = {}
        
        # Check if it's a stored reference or catalog item
        if saree_id in reference_sarees:
            saree_emb = reference_sarees[saree_id]["embedding"]
            saree_meta = {"palette": reference_sarees[saree_id]["palette"]}
        else:
            # Look up in catalog
            saree_row = catalog_df[catalog_df["item_id"] == saree_id]
            if not saree_row.empty:
                saree_idx = int(saree_row.iloc[0]["embedding_index"])
                saree_emb = embeddings[saree_idx]
                saree_meta = saree_row.iloc[0].to_dict()
        
        if saree_emb is None:
            raise HTTPException(status_code=404, detail=f"Saree '{saree_id}' not found")
        
        # Get blouse embedding
        blouse_row = catalog_df[catalog_df["item_id"] == blouse_id]
        if blouse_row.empty:
            raise HTTPException(status_code=404, detail=f"Blouse '{blouse_id}' not found")
        
        blouse_idx = int(blouse_row.iloc[0]["embedding_index"])
        blouse_emb = embeddings[blouse_idx]
        blouse_meta = blouse_row.iloc[0].to_dict()
        
        # Add to training data
        score = 1.0 if is_good_match else 0.0
        trainer.add_training_pair(
            saree_id=saree_id,
            blouse_id=blouse_id,
            saree_embedding=saree_emb,
            blouse_embedding=blouse_emb,
            saree_meta=saree_meta,
            blouse_meta=blouse_meta,
            score=score
        )
        
        return {
            "status": "success",
            "message": f"Training updated with {'positive' if is_good_match else 'negative'} example",
            "total_training_pairs": len(trainer.training_pairs)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training_stats")
def get_training_stats():
    """Get statistics about the training data."""
    if trainer is None:
        return {"error": "Training system not initialized"}
    
    stats = trainer.get_training_stats()
    return {
        "total_pairs": stats["total_pairs"],
        "avg_score": round(stats["avg_score"], 3),
        "color_combinations": stats["color_pairs"],
        "most_common_pair": stats.get("most_common"),
        "model_weights": {
            "image_similarity": ALPHA,
            "metadata": BETA,
            "training_boost": GAMMA
        }
    }


@app.get("/suggest_colors/{saree_color}")
def suggest_compatible_colors(saree_color: str):
    """
    Suggest compatible blouse colors for a given saree color based on training data.
    """
    if trainer is None or len(trainer.training_pairs) == 0:
        return {
            "saree_color": saree_color,
            "suggestions": [],
            "message": "No training data available yet"
        }
    
    suggestions = trainer.suggest_color_compatibility(saree_color)
    
    return {
        "saree_color": saree_color,
        "suggestions": [
            {"blouse_color": color, "confidence": round(conf, 3)}
            for color, conf in suggestions
        ]
    }