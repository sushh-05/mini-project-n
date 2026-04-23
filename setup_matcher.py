from pathlib import Path
import textwrap

PROJECT_NAME = "saree-blouse-matcher"

folders = [
    "data/images/sarees",
    "data/images/blouses",
    "models",
    "outputs",
    "scripts",
    "notebooks",
]

requirements_txt = """
pandas
numpy
scikit-learn
pillow
torch
torchvision
transformers
tqdm
"""

readme_md = """
# Saree Blouse Matcher

Hybrid recommendation MVP:
- image embeddings for sarees and blouses
- cosine similarity retrieval
- metadata-based reranking
"""

catalog_csv = """
item_id,item_type,image_path,color,fabric,pattern,occasion,style
S001,saree,data/images/sarees/s001.jpg,red,silk,zari,festive,traditional
S002,saree,data/images/sarees/s002.jpg,blue,georgette,printed,party,modern
B001,blouse,data/images/blouses/b001.jpg,gold,silk,plain,festive,traditional
B002,blouse,data/images/blouses/b002.jpg,green,silk,embroidered,festive,traditional
B003,blouse,data/images/blouses/b003.jpg,silver,satin,plain,party,modern
"""

extract_embeddings_py = """
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "data" / "catalog.csv"
    output_path = project_root / "outputs" / "embeddings.npy"

    df = pd.read_csv(catalog_path)

    # Placeholder random embeddings for initial wiring test
    # Replace this with CLIP / timm embedding extraction next
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(len(df), 512)).astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    print(f"Saved dummy embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
"""

recommend_py = """
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ALPHA = 0.7
BETA = 0.3

def exact_match(a, b):
    return 1.0 if str(a).strip().lower() == str(b).strip().lower() else 0.0

def color_score(saree_color, blouse_color):
    saree_color = str(saree_color).strip().lower()
    blouse_color = str(blouse_color).strip().lower()

    if saree_color == blouse_color:
        return 1.0

    contrast_pairs = {
        ("red", "gold"),
        ("green", "gold"),
        ("blue", "silver"),
        ("maroon", "gold"),
        ("pink", "green"),
        ("black", "red"),
    }

    if (saree_color, blouse_color) in contrast_pairs or (blouse_color, saree_color) in contrast_pairs:
        return 0.85

    return 0.3

def attribute_score(saree_row, blouse_row):
    c = color_score(saree_row["color"], blouse_row["color"])
    f = exact_match(saree_row["fabric"], blouse_row["fabric"])
    p = exact_match(saree_row["pattern"], blouse_row["pattern"])
    o = exact_match(saree_row["occasion"], blouse_row["occasion"])
    s = exact_match(saree_row["style"], blouse_row["style"])
    return 0.35 * c + 0.20 * f + 0.15 * p + 0.15 * o + 0.15 * s

def get_matching_blouses(saree_id, df, embeddings, top_k=5, candidate_k=20):
    saree_idx = df.index[df["item_id"] == saree_id][0]
    saree_vec = embeddings[saree_idx].reshape(1, -1)

    blouse_df = df[df["item_type"] == "blouse"].copy()
    blouse_indices = blouse_df.index.to_list()
    blouse_vectors = embeddings[blouse_indices]

    sims = cosine_similarity(saree_vec, blouse_vectors)[0]
    blouse_df["image_score"] = sims

    saree_row = df.loc[saree_idx]
    blouse_df = blouse_df.sort_values("image_score", ascending=False).head(candidate_k).copy()
    blouse_df["attr_score"] = blouse_df.apply(lambda row: attribute_score(saree_row, row), axis=1)
    blouse_df["final_score"] = ALPHA * blouse_df["image_score"] + BETA * blouse_df["attr_score"]

    return blouse_df.sort_values("final_score", ascending=False).head(top_k)

def main():
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "data" / "catalog.csv"
    embeddings_path = project_root / "outputs" / "embeddings.npy"

    df = pd.read_csv(catalog_path)
    embeddings = np.load(embeddings_path)

    result = get_matching_blouses("S001", df, embeddings, top_k=3)
    print(result[["item_id", "image_path", "image_score", "attr_score", "final_score"]])

if __name__ == "__main__":
    main()
"""

gitignore_txt = """
.venv/
__pycache__/
*.pyc
outputs/*.npy
"""

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
        print(f"Created: {path}")
    else:
        print(f"Skipped (already exists): {path}")

def main():
    root = Path(PROJECT_NAME)
    root.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        folder_path = root / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")

    write_file(root / "requirements.txt", requirements_txt)
    write_file(root / "README.md", readme_md)
    write_file(root / ".gitignore", gitignore_txt)
    write_file(root / "data" / "catalog.csv", catalog_csv)
    write_file(root / "scripts" / "extract_embeddings.py", extract_embeddings_py)
    write_file(root / "scripts" / "recommend.py", recommend_py)

    print("\\nSetup complete.")
    print(f"\\nNext steps:")
    print(f"1. cd {PROJECT_NAME}")
    print("2. python -m venv .venv")
    print("3. source .venv/bin/activate")
    print("4. pip install -r requirements.txt")
    print("5. python scripts/extract_embeddings.py")
    print("6. python scripts/recommend.py")

if __name__ == "__main__":
    main()
