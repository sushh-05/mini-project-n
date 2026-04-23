from pathlib import Path
from io import BytesIO
import colorsys
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
ALPHA = 0.7
BETA = 0.3

app = FastAPI(title="Saree Blouse Matcher API")

COLOR_MATCH_SCORES = {
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
        features = model.get_image_features(**inputs)

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

    neutral_colors = {"gold", "silver", "black", "white", "beige", "cream"}
    if b in neutral_colors:
        return 0.75

    return 0.25

def palette_color_score(query_palette, blouse_color):
    blouse_color = normalize_text(blouse_color)
    if not blouse_color or not query_palette:
        return 0.0

    return float(sum(pair_color_score(p["name"], blouse_color) * p["share"] for p in query_palette))

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
    c = palette_color_score(query_meta.get("palette", []), blouse_row.get("color"))
    f = fabric_score(query_meta.get("fabric"), blouse_row.get("fabric"))
    p = pattern_score(query_meta.get("pattern"), blouse_row.get("pattern"))
    o = occasion_score(query_meta.get("occasion"), blouse_row.get("occasion"))
    s = style_score(query_meta.get("style"), blouse_row.get("style"))

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
    result_df["final_score"] = ALPHA * result_df["image_score"] + BETA * result_df["attr_score"]

    result_df = result_df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
    return result_df

@app.on_event("startup")
def startup_event():
    global model, processor, catalog_df, embeddings

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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": MODEL_NAME,
        "catalog_rows": int(len(catalog_df)) if catalog_df is not None else 0
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