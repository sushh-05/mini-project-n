"""
Create composite image from multiple blouse images for testing object detection.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse


def create_composite_grid(image_paths, output_path, grid_cols=3, img_size=(300, 300)):
    """
    Create a grid composite image from multiple images.
    
    Args:
        image_paths: List of image file paths
        output_path: Where to save the composite
        grid_cols: Number of columns in grid
        img_size: Size to resize each image to (width, height)
    
    Returns:
        Dictionary with grid information (positions, item_ids, etc.)
    """
    if not image_paths:
        raise ValueError("No images provided")
    
    # Load and resize images
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    
    if not images:
        raise ValueError("No valid images found")
    
    # Calculate grid dimensions
    n_images = len(images)
    n_cols = min(grid_cols, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create composite canvas
    canvas_width = n_cols * img_size[0]
    canvas_height = n_rows * img_size[1]
    composite = Image.new("RGB", (canvas_width, canvas_height), color=(240, 240, 240))
    
    # Store position information
    grid_info = {
        "grid_cols": n_cols,
        "grid_rows": n_rows,
        "img_size": img_size,
        "items": []
    }
    
    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        x = col * img_size[0]
        y = row * img_size[1]
        
        composite.paste(img, (x, y))
        
        # Store bounding box info
        grid_info["items"].append({
            "index": idx,
            "path": str(valid_paths[idx]),
            "bbox": [x, y, x + img_size[0], y + img_size[1]]  # [x1, y1, x2, y2]
        })
    
    # Save composite
    composite.save(output_path, quality=95)
    print(f"✓ Created composite image: {output_path}")
    print(f"  Grid: {n_rows} rows × {n_cols} cols")
    print(f"  Total images: {n_images}")
    
    return grid_info


def main():
    parser = argparse.ArgumentParser(description="Create composite image from blouse images")
    parser.add_argument("--catalog", type=str, default="data/catalog.csv", 
                       help="Path to catalog CSV")
    parser.add_argument("--output", type=str, default="data/composite_blouses.jpg",
                       help="Output composite image path")
    parser.add_argument("--cols", type=int, default=3,
                       help="Number of columns in grid")
    parser.add_argument("--size", type=int, default=300,
                       help="Size of each image in pixels (square)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / args.catalog
    output_path = project_root / args.output
    
    # Read catalog and get blouse images
    import pandas as pd
    df = pd.read_csv(catalog_path)
    blouse_df = df[df["item_type"].str.lower() == "blouse"]
    
    if blouse_df.empty:
        print("No blouse items found in catalog!")
        return
    
    image_paths = []
    for _, row in blouse_df.iterrows():
        img_path = project_root / row["image_path"]
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if not image_paths:
        print("No valid blouse images found!")
        return
    
    # Create composite
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_info = create_composite_grid(
        image_paths,
        output_path,
        grid_cols=args.cols,
        img_size=(args.size, args.size)
    )
    
    # Save grid info
    import json
    info_path = output_path.parent / (output_path.stem + "_grid_info.json")
    with open(info_path, "w") as f:
        json.dump(grid_info, f, indent=2)
    print(f"✓ Saved grid info: {info_path}")


if __name__ == "__main__":
    main()
