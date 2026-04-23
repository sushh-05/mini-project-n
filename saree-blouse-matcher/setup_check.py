"""
Setup script to prepare the project for first-time use.
Creates sample images if none exist.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np


def create_sample_image(path, color, text, size=(400, 400)):
    """Create a colored sample image with text."""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Draw text in center
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw shadow
    draw.text((x+2, y+2), text, fill=(0, 0, 0), font=font)
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"✓ Created: {path}")


def check_setup():
    """Check current project setup status."""
    project_root = Path(__file__).resolve().parent
    
    print("=" * 60)
    print("Saree Blouse Matcher - Setup Check")
    print("=" * 60)
    
    # Check catalog
    catalog_path = project_root / "data" / "catalog.csv"
    if catalog_path.exists():
        df = pd.read_csv(catalog_path)
        print(f"\n✓ Catalog found: {len(df)} items")
        print(f"  - Sarees: {len(df[df['item_type']=='saree'])}")
        print(f"  - Blouses: {len(df[df['item_type']=='blouse'])}")
    else:
        print("\n✗ Catalog not found")
        return False
    
    # Check images
    missing_images = []
    for _, row in df.iterrows():
        img_path = project_root / row["image_path"]
        if not img_path.exists():
            missing_images.append(img_path)
    
    if missing_images:
        print(f"\n⚠ Missing {len(missing_images)} images:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")
        return False
    else:
        print(f"\n✓ All {len(df)} images found")
    
    # Check outputs
    embeddings_path = project_root / "outputs" / "embeddings.npy"
    catalog_status_path = project_root / "outputs" / "catalog_with_status.csv"
    
    if embeddings_path.exists():
        emb = np.load(embeddings_path)
        print(f"\n✓ Embeddings found: {emb.shape}")
    else:
        print("\n⚠ Embeddings not generated yet")
        print("  Run: python scripts/extract_embeddings.py")
    
    if catalog_status_path.exists():
        print(f"✓ Catalog status found")
    else:
        print("⚠ Catalog status not generated yet")
    
    return True


def create_sample_data():
    """Create sample images for testing if they don't exist."""
    project_root = Path(__file__).resolve().parent
    
    print("\n" + "=" * 60)
    print("Creating Sample Data")
    print("=" * 60)
    
    # Sample data configuration
    samples = {
        "data/images/sarees/s001.jpg": ((200, 50, 60), "Red Silk Saree"),
        "data/images/sarees/s002.jpg": ((60, 90, 200), "Blue Georgette Saree"),
        "data/images/blouses/b001.jpg": ((212, 175, 55), "Gold Silk Blouse"),
        "data/images/blouses/b002.jpg": ((50, 140, 70), "Green Silk Blouse"),
        "data/images/blouses/b003.jpg": ((180, 180, 190), "Silver Satin Blouse"),
    }
    
    created = 0
    for rel_path, (color, text) in samples.items():
        full_path = project_root / rel_path
        if not full_path.exists():
            create_sample_image(full_path, color, text)
            created += 1
        else:
            print(f"  Skipped (exists): {full_path}")
    
    print(f"\n✓ Created {created} sample images")
    
    if created > 0:
        print("\n⚠ These are placeholder images for testing!")
        print("  Replace them with real saree/blouse images for actual use.")


def main():
    project_root = Path(__file__).resolve().parent
    
    print("\n🎯 Saree Blouse Matcher - First Time Setup\n")
    
    # Check if images exist
    catalog_path = project_root / "data" / "catalog.csv"
    if not catalog_path.exists():
        print("✗ catalog.csv not found!")
        print("  Please create data/catalog.csv first")
        return
    
    df = pd.read_csv(catalog_path)
    images_exist = all((project_root / row["image_path"]).exists() for _, row in df.iterrows())
    
    if not images_exist:
        response = input("\nImages are missing. Create sample images for testing? (y/n): ")
        if response.lower() == 'y':
            create_sample_data()
        else:
            print("\n⚠ Please add your images before proceeding:")
            print("  1. Place images in data/images/sarees/ and data/images/blouses/")
            print("  2. Update paths in data/catalog.csv")
            return
    
    # Final check
    print("\n")
    if check_setup():
        print("\n" + "=" * 60)
        print("✓ Setup Complete! Next steps:")
        print("=" * 60)
        print("1. Extract embeddings: python scripts/extract_embeddings.py")
        print("2. Start API: uvicorn api:app --reload")
        print("3. Open frontend.html in browser")
        print("=" * 60)
    else:
        print("\n⚠ Setup incomplete. Please fix the issues above.")


if __name__ == "__main__":
    main()
