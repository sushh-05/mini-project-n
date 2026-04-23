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
from tqdm import tqdm

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8
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
    "crple": np.array([130, 70, 170]),
    "yerange": np.array([230, 130, 40]),
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

def extract_color_palette_info(image_path: Path, n_clusters: int = 5, resize_to=(180, 180)):
    image = load_image(image_path)
    image = image.resize(resize_to)
    pixels = np.array(image).reshape(-1, 3)

    filtered_pixels = filter_background_pixels(pixels)

    n_clusters = min(n_clusters, max(1, len(filtered_pixels)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(filtered_pixels)
    centers = kmeans.cluster_centers_

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    palette = []
    total = counts.sum()

    for idx in order:
        rgb = centers[idx].astype(int)
        share = float(counts[idx] / total)
        palette.append({
            "rgb": rgb,
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "name": closest_color_name(rgb),
            "share": round(share, 4),
        })

    return {
        "palette": palette,
        "total_pixels": int(len(pixels)),
        "filtered_pixels": int(len(filtered_pixels)),
    }

def embed_images(images, model, processor, device: str) -> np.ndarray:
    all_embeddings = []

    for start in tqdm(range(0, len(images), BATCH_SIZE), desc="Embedding catalog"):
        batch_images = images[start:start + BATCH_SIZE]
        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = model.get_image_features(**inputs)

        batch_embeddings = image_features.cpu().numpy()
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")
    return l2_normalize(embeddings)

def embed_single_image(image_path: Path, model, processor, device: str) -> np.ndarray:
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

    emb = features.cpu().numpy().astype("float32")
    return l2_normalize(emb)

def build_or_load_catalog_embeddings(project_root: Path, model, processor, device: str):
    catalog_path = project_root / "data" / "catalog.csv"
    output_dir = project_root / "outputs"
    embeddings_path = output_dir / "embeddings.npy"
    enriched_catalog_path = output_dir / "catalog_with_status.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    if embeddings_path.exists() and enriched_catalog_path.exists():
        df = pd.read_csv(enriched_catalog_path)
        embeddings = np.load(embeddings_path)
        return df, embeddings

    df = pd.read_csv(catalog_path)
    valid_rows = []
    images = []

    for idx, row in df.iterrows():
        image_path = project_root / row["image_path"]
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue
        try:
            images.append(load_image(image_path))
            valid_rows.append(idx)
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")

    if not images:
        raise RuntimeError("No valid catalog images found.")

    embeddings = embed_images(images, model, processor, device)

    valid_df = df.loc[valid_rows].reset_index(drop=True).copy()
    valid_df["embedding_index"] = np.arange(len(valid_df))
    valid_df["embedding_model"] = MODEL_NAME

    np.save(embeddings_path, embeddings)
    valid_df.to_csv(enriched_catalog_path, index=False)

    return valid_df, embeddings

def exact_match(a, b):
    return 1.0 if normalize_text(a) == normalize_text(b) and normalize_text(a) !rple": np.array([130, 70, 170]),
    "ye    a = normalize_text(a)
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
    if not query_palette:
        return 0.0

    blouse_color = normalize_text(blouse_color)
    if not blouse_color:
        return 0.0

    weighted = []
    for item in query_palette:
        score = pair_color_score(item["name"], blouse_color)
        weighted.append(score * itemrple": np.array([130, 70, 170]),
    "yed))

def fabric_score(query_fabric, blouse_fabric):
    query_fabric = normalize_text(query_fabric)
    blouse_fabric = normalize_text(blouse_fabric)

    if not query_fabric or not blouse_fabric:
        return 0.0
    if query_fabric == blouse_farple": np.array([130, 70, 170]),
    "ye = {
        ("silk", "brocade"),
        ("silk", "silk"),
        ("cotton", "cotton"),
        ("georgette", "satirple": np.array([130, 70, 170]),
    "ye    ("chiffon", "satirple": np.array([130, 70, 170]),
    "ye  if (query_fabric, blouse_fabric) in compatible or (blourple": np.array([130, 70, 170]),
    "ye        return 0.8

    return 0.3

def pattern_score(query_pattern, blouse_pattern):
    query_pattern = normalize_rple": np.array([130, 70, 170]),
    "ye normalize_text(blouse_pattern)

    if not qrple": np.array([130, 70, 170]),
    "ye    return 0.0
    if query_pattern == blouse_patrple": np.array([130, 70, 170]),
    "yettern in {"zari", "embroidered"} and blouse_pattern in {"plain", "brocade"}:
        return 0.85
    if query_patrple": np.array([130, 70, 170]),
    "yen {"plain"}:
        return 0.9
    if query_pattern in {"plain"} and blouse_pattern in {"embroidered", "zari"}:
        return 0.85
    return rple": np.array([130, 70, 170]),
    "ye blouse_occasion):
    return max(exact_match(query_occarple": np.array([130, 70, 170]),
    "yeery_occasion, blouse_occasion))

def style_score(query_style, blouse_style):
    return max(exact_match(query_srple": np.array([130, 70, 170]),
    "ye_style, blouse_style))

def attribute_score(query_meta: dict, blouse_row):
    c = palette_color_score(query_rple": np.array([130, 70, 170]),
    "ye"color"))
    f = fabric_srple": np.array([130, 70, 170]),
    "yew.get("fabric"))
    p = pattern_srple": np.array([130, 70, 170]),
    "yeow.get("pattern"))
    o = occarple": np.array([130, 70, 170]),
    "yelouse_row.get("occasion"))
    s = style_score(query_meta.get("style"), blouse_rowrple": np.array([130, 70, 170]),
    "yec + 0.20 * f + rple": np.array([130, 70, 170]),
    "yern final_attr, {
        "palette_crple": np.array([130, 70, 170]),
    "yef,
        "pattern_score": p,
        "occasion_score": o,
        "style_score": s,
    }

def recommend_from_query(query_embedding, blouse_df, embeddrple": np.array([130, 70, 170]),
    "ye0):
    blouse_embedding_indices = blouse_dfrple": np.array([130, 70, 170]),
    "ye()
    blouse_vectors = embeddings[blouse_embedding_indices]

    sims = cosine_similarity(query_embedding, blouse_vectorrple": np.array([130, 70, 170]),
    "ye   result_df["image_score"] = sims
    result_df = resulrple": np.array([130, 70, 170]),
    "yeg=False).head(candidate_k).copy()

    attr_scores = []
    breakrple": np.array([130, 70, 170]),
    "yeiterrows():
        score, breakdown = attribute_score(query_meta,rple": np.array([130, 70, 170]),
    "ye        breakdowns.append(breakdown)

    result_rple": np.array([130, 70, 170]),
    "yet_df["palette_color_score"] rple": np.array([130, 70, 170]),
    "yeakdowns]
    result_df["fabric_score"] = [b["fabric_score"] for b in breakdorple": np.array([130, 70, 170]),
    "ye["pattern_score"] for b in breakdorple": np.array([130, 70, 170]),
    "yee": np.array([130, 70, 170]),
    "yens]
    result_rple": np.array([130, 70, 170]),
    "yer b in breakrple": np.array([130, 70, 170]),
    "yePHA * rerple": np.array([130, 70, 170]),
    "yef["attr_score"]

    return result_df.sort_values("final_score", ascending=False).head(top_k).reset_irple": np.array([130, 70, 170]),
    "ye= argparse.ArgumentParser(description="End-to-end saree to blouse recommendarple": np.array([130, 70, 170]),
    "ye("--image_path", type=str, required=True, help="Path to query saree image")
    parserrple": np.array([130, 70, 170]),
    "yelt=5)
    parser.add_argument("--candidate_k", type=int, defaultrple": np.array([130, 70, 170]),
    "yek", type=int, default=5)
    parple": np.array([130, 70, 170]),
    "yedefault="")
    parser.add_argument("--patterple": np.array([130, 70, 170]),
    "yed_argument("--occasion", type=str, defrple": np.array([130, 70, 170]),
    "yee", type=str, default="")
    args = parser.prple": np.array([130, 70, 170]),
    "yeile__).resolve().parents[1]
    query_image_path = Path(args.image_prple": np.array([130, 70, 170]),
    "yelute():
        query_image_path = project_root / query_image_path

    if not qrple": np.array([130, 70, 170]),
    "yeFileNotFoundError(f"Query image not found: {query_image_path}")

    device = "cuda" if torch.cuda.is_available() rple": np.array([130, 70, 170]),
    "yevice}")
    print(f"Loading model: {MODEL_NAME}")

    model = CLIPMrple": np.array([130, 70, 170]),
    "yece)
    processor = CLIPProcessor.from_pretrarple": np.array([130, 70, 170]),
    "yef, embeddrple": np.array([130, 70, 170]),
    "yeproject_root, morple": np.array([130, 70, 170]),
    "yedfrple": np.array([130, 70, 170]),
    "yeay([130, 70, 170]),
    "yempty:
        raise RuntimeError("No blrple": np.array([130, 70, 170]),
    "yey_embedding = embed_sirple": np.array([130, 70, 170]),
    "yearray([130, 70, 170]),
    "yeo = extrple": np.array([130, 70, 170]),
    "ye, n_clusters=args.palette_k)

    query_meta =rple": np.array([130, 70, 170]),
    "yette"rple": np.array([130, 70, 170]),
    "ye  "pattern": args.pattern,
        "occasion": args.occasion,
        "style": args.style,
    }

    recommendatrple": np.array([130, 70, 170]),
    "yery_embedrple": np.array([130, 70, 170]),
    "yeblouserple": np.array([130, 70, 170]),
    "ye    query_meta=query_meta,
        top_k=args.top_k,
        candirple": np.array([130, 70, 170]),
    "yeple": np.array([130, 70, 170]),
    "ye_info["palette"]]
    palette_hexes = [p["hex"] for p in palette_inrple": np.array([130, 70, 170]),
    "yeshare"] for p in palette_info["palette"]]

    recommendations["query_palette_names"] = str(palette_names)
    recommendations["query_palette_hexes"] = str(palette_hexes)
    recommendations["query_palette_shares"] = str(palette_shares)
    recommendations["query_total_pixels"] = palette_info["total_pixels"]
    recommendations["query_filtered_pixels"] = palette_info["filtered_pixels"]

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"pipeline_results_{query_image_path.stem}.csv"
    recommendations.to_csv(results_path, index=False)

    print("\nDetected query palette:")
    for i, p in enumerate(palette_info["palette"], start=1):
        print(f"{i}. {p['name']:>7} | {p['hex']} | share={p['share']}")

    show_cols = [
        "item_id", "image_path", "color", "fabric", "pattern", "occasion", "style",
        "image_score", "palette_color_score", "attr_score", "final_score"
    ]
    show_cols = [c for c in show_cols if c in recommendations.columns]

    print("\nTop recommendations:\n")
    print(recommendations[show_cols].to_string(index=False))
    print(f"\nSaved results to: {results_path}")

if __name__ == "__main__":
    main()