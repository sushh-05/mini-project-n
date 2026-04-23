from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 8

def load_image(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    return image

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms

def main():
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "data" / "catalog.csv"
    output_dir = project_root / "outputs"
    embeddings_path = output_dir / "embeddings.npy"
    enriched_catalog_path = output_dir / "catalog_with_status.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog_path)
    if "image_path" not in df.columns:
        raise ValueError("catalog.csv must contain an 'image_path' column")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    valid_rows = []
    images = []

    for idx, row in df.iterrows():
        image_path = project_root / row["image_path"]
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            img = load_image(image_path)
            images.append(img)
            valid_rows.append(idx)
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")

    if not images:
        raise RuntimeError("No valid images found. Please check your image paths in catalog.csv")

    all_embeddings = []

    for start in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting embeddings"):
        batch_images = images[start:start + BATCH_SIZE]

        inputs = processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            image_features = model.get_image_features(**inputs)

        batch_embeddings = image_features.cpu().numpy()
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")
    embeddings = l2_normalize(embeddings)

    valid_df = df.loc[valid_rows].reset_index(drop=True).copy()
    valid_df["embedding_index"] = np.arange(len(valid_df))
    valid_df["embedding_model"] = MODEL_NAME

    np.save(embeddings_path, embeddings)
    valid_df.to_csv(enriched_catalog_path, index=False)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved filtered catalog to: {enriched_catalog_path}")
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()