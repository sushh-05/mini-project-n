"""
Train YOLO model for blouse detection.
This script prepares data and trains a YOLOv8 model to detect blouses.
"""
from pathlib import Path
import yaml
import shutil
from ultralytics import YOLO


def create_yolo_dataset_structure(project_root):
    """Create YOLO dataset folder structure."""
    dataset_root = project_root / "yolo_dataset"
    
    folders = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    
    return dataset_root


def create_dataset_yaml(dataset_root, num_classes=1):
    """Create YOLO dataset configuration file."""
    config = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'blouse'
        }
    }
    
    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Created dataset config: {yaml_path}")
    return yaml_path


def train_yolo_model(dataset_yaml, epochs=50, model_size='n'):
    """
    Train YOLO model.
    
    Args:
        dataset_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    """
    print(f"\nTraining YOLOv8{model_size} for {epochs} epochs...")
    print("="*60)
    
    # Load pretrained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        name='blouse_detector',
        project='yolo_training',
        patience=10,  # Early stopping
        save=True,
        save_period=10,  # Save every 10 epochs
        device='0' if YOLO.cuda.is_available() else 'cpu',
        workers=4,
        verbose=True
    )
    
    print("\n✓ Training completed!")
    print(f"Best model saved to: yolo_training/blouse_detector/weights/best.pt")
    
    return results


def validate_model(model_path, dataset_yaml):
    """Validate trained model."""
    print("\nValidating model...")
    model = YOLO(model_path)
    metrics = model.val(data=dataset_yaml)
    
    print("\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    return metrics


def main():
    """Main training pipeline."""
    project_root = Path(__file__).resolve().parents[1]
    
    print("="*60)
    print("YOLO Blouse Detector Training")
    print("="*60)
    
    # Check if labeled data exists
    print("\nStep 1: Checking for labeled data...")
    dataset_root = project_root / "yolo_dataset"
    
    if not dataset_root.exists():
        print("\n⚠️  No dataset found!")
        print("\nTo train YOLO, you need:")
        print("1. Images of scenes with blouses")
        print("2. YOLO format labels (class x_center y_center width height)")
        print("\nOptions:")
        print("  A. Download DeepFashion dataset")
        print("  B. Label your own images with LabelImg or Roboflow")
        print("  C. Use pretrained fashion detection model")
        print("\nSee DATASETS.md for details.")
        return
    
    # Count training images
    train_images = list((dataset_root / "images" / "train").glob("*.*"))
    val_images = list((dataset_root / "images" / "val").glob("*.*"))
    
    if not train_images:
        print("\n⚠️  No training images found in yolo_dataset/images/train/")
        print("Please add labeled images before training.")
        return
    
    print(f"✓ Found {len(train_images)} training images")
    print(f"✓ Found {len(val_images)} validation images")
    
    # Create dataset YAML
    print("\nStep 2: Creating dataset configuration...")
    dataset_yaml = create_dataset_yaml(dataset_root)
    
    # Train model
    print("\nStep 3: Training model...")
    print("This may take 2-6 hours depending on dataset size and hardware.")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    results = train_yolo_model(dataset_yaml, epochs=50, model_size='n')
    
    # Validate
    print("\nStep 4: Validating model...")
    best_model = project_root / "yolo_training" / "blouse_detector" / "weights" / "best.pt"
    
    if best_model.exists():
        metrics = validate_model(best_model, dataset_yaml)
        
        # Copy to models directory
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        final_path = models_dir / "blouse_detector.pt"
        shutil.copy(best_model, final_path)
        
        print(f"\n✓ Model copied to: {final_path}")
        print("\nThe API will now use this custom model automatically!")
    else:
        print("\n⚠️  Training completed but best model not found.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
