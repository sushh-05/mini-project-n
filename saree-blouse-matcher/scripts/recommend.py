from pathlib import Path
import argparse
import colorsys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
ALPHA = 0.7
BETA = 0.3

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

def embed_single_image(image_path: Path, model, processor, device: str) -> np.ndarray:
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

    emb = features.cpu().numpy().astype("float32")
    emb = l2_normalize(emb)
    return emb

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

def extract_dominant_color_info(image_path: Path, n_clusters: int = 4, resize_to=(180, 180)):
    image = load_image(image_path)
    image = image.resize(resize_to)
    pixels = np.array(image).reshape(-1, 3)

    filtered_pixels = filter_background_pixels(pixels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(filtered_pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    dominant_idx = np.argmax(counts)
    dominant_rgb = centers[dominant_idx].astype(int)

    color_name = closest_color_name(dominant_rgb)

    return {
        "dominant_rgb": dominant_rgb,
        "dominant_hex": "#{:02x}{:02x}{:02x}".format(*dominant_rgb),
        "dominant_color_name": color_name,
        "total_pixels": int(len(pixels)),
        "filtered_pixels": int(len(filtered_pixels)),
    }

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

def color_score(query_color, blouse_color):
    query_color = normalize_text(query_color)
    blouse_color = normalize_text(blouse_color)

    if not query_color or not blouse_color:
        return 0.0

    if query_color == blouse_color:
        return 1.0

    if (query_color, blouse_color) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(query_color, blouse_color)]

    if (blouse_color, query_color) in COLOR_MATCH_SCORES:
        return COLOR_MATCH_SCORES[(blouse_color, query_color)]

    neutral_colors = {"gold", "silver", "black", "white", "beige", "cream"}
    if blouse_color in neutral_colors:
        return 0.75

    return 0.25

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
    c = color_score(query_meta.get("color"), blouse_row.get("color"))
    f = fabric_score(query_meta.get("fabric"), blouse_row.get("fabric"))
    p = pattern_score(query_meta.get("pattern"), blouse_row.get("pattern"))
    o = occasion_score(query_meta.get("occasion"), blouse_row.get("occasion"))
    s = style_score(query_meta.get("style"), blouse_row.get("style"))

    final_attr = (
        0.35 * c +
        0.20 * f +
        0.15 * p +
        0.15 * o +
        0.15 * s
    )

    return final_attr, {
        "color_score": c,
        "fabric_score": f,
        "pattern_score": p,
        "occasion_score": o,
        "style_score": s,
    }

def recommend_from_image(query_embedding, blouse_df, embeddings, query_meta=None, top_k=5, candidate_k=20):
    blouse_embedding_indices = blouse_df["embedding_index"].astype(int).to_numpy()
    blouse_vectors = embeddings[blouse_embedding_indices]

    sims = cosine_similarity(query_embedding, blouse_vectors)[0]
    result_df = blouse_df.copy()
    result_df["image_score"] = sims
    result_df = result_df.sort_values("image_score", ascending=False).head(candidate_k).copy()

    if query_meta:
        attr_scores = []
        breakdowns = []

        for _, row in result_df.iterrows():
            score, breakdown = attribute_score(query_meta, row)
            attr_scores.append(score)
            breakdowns.append(breakdown)

        result_df["attr_score"] = attr_scores
        result_df["color_score"] = [b["color_score"] for b in breakdowns]
        result_df["fabric_score"] = [b["fabric_score"] for b in breakdowns]
        result_df["pattern_score"] = [b["pattern_score"] for b in breakdowns]
        result_df["occasion_score"] = [b["occasion_score"] for b in breakdowns]
        result_df["style_score"] = [b["style_score"] for b in breakdowns]
        result_df["final_score"] = ALPHA * result_df["image_score"] + BETA * result_df["attr_score"]
    else:
        result_df["attr_score"] = 0.0
        result_df["color_score"] = 0.0
        result_df["fabric_score"] = 0.0
        result_df["pattern_score"] = 0.0
        result_df["occasion_score"] = 0.0
        result_df["style_score"] = 0.0
        result_df["final_score"] = result_df["image_score"]

    result_df = result_df.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Recommend matching blouses from a saree image with background-filtered dominant color extraction.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to query saree image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of blouse recommendations to return")
    parser.add_argument("--candidate_k", type=int, default=20, help="Top candidates from image similarity before reranking")

    parser.add_argument("--fabric", type=str, default="", help="Optional query saree fabric")
    parser.add_argument("--pattern", type=str, default="", help="Optional query saree pattern")
    parser.add_argument("--occasion", type=str, default="", help="Optional query saree occasion")
    parser.add_argument("--style", type=str, default="", help="Optional query saree style")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "outputs" / "catalog_with_status.csv"
    embeddings_path = project_root / "outputs" / "embeddings.npy"
    query_image_path = Path(args.image_path)

    if not query_image_path.is_absolute():
        query_image_path = project_root / query_image_path

    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing file: {catalog_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing file: {embeddings_path}")

    df = pd.read_csv(catalog_path)
    embeddings = np.load(embeddings_path)

    blouse_df = df[df["item_type"].str.lower() == "blouse"].copy()
    if blouse_df.empty:
        raise RuntimeError("No blouse items found in catalog_with_status.csv")

    color_info = extract_dominant_color_info(query_image_path)

    query_meta = {
        "color": color_info["dominant_color_name"],
        "fabric": args.fabric,
        "pattern": args.pattern,
        "occasion": args.occasion,
        "style": args.style,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    query_embedding = embed_single_image(query_image_path, model, processor, device)

    recommendations = recommend_from_image(
        query_embedding=query_embedding,
        blouse_df=blouse_df,
        embeddings=embeddings,
        query_meta=query_meta,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    recommendations["query_detected_color"] = color_info["dominant_color_name"]
    recommendations["query_dominant_rgb"] = str(tuple(color_info["dominant_rgb"].tolist()))
    recommendations["query_dominant_hex"] = color_info["dominant_hex"]
    recommendations["query_total_pixels"] = color_info["total_pixels"]
    recommendations["query_filtered_pixels"] = color_info["filtered_pixels"]

    output_name = f"recommendations_from_{query_image_path.stem}.csv"
    results_path = project_root / "outputs" / output_name
    recommendations.to_csv(results_path, index=False)

    print("\nDetected query color:")
    print(f"  Name          : {color_info['dominant_color_name']}")
    print(f"  RGB           : {tuple(color_info['dominant_rgb'].tolist())}")
    print(f"  HEX           : {color_info['dominant_hex']}")
    print(f"  Total pixels  : {color_info['total_pixels']}")
    print(f"  Filtered pixels used: {color_info['filtered_pixels']}")

    columns_to_show = [
        "item_id", "image_path", "color", "fabric", "pattern", "occasion", "style",
        "image_score", "attr_score", "final_score"
    ]
    available_columns = [c for c in columns_to_show if c in recommendations.columns]

    print("\nTop recommendations:\n")
    print(recommendations[available_columns].to_string(index=False))
    print(f"\nSaved recommendations to: {results_path}")

if __name__ == "__main__":
    main()