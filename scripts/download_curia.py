#!/usr/bin/env python3
"""
Script to download raidium/curia model from HuggingFace and save it locally.

This allows for faster loading and offline usage.
"""

from pathlib import Path

from transformers import AutoImageProcessor, AutoModel


def download_curia_model():
    """Download CURIA model from HuggingFace to local pretrain directory."""

    # Define paths
    project_root = Path(__file__).parent.parent
    pretrain_dir = project_root / "pretrain" / "raidium" / "curia"

    print(f"Downloading raidium/curia model...")
    print(f"Target directory: {pretrain_dir}")

    # Create directory
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    # Download processor and model
    print("\n1. Downloading image processor...")
    processor = AutoImageProcessor.from_pretrained("raidium/curia")
    processor.save_pretrained(pretrain_dir)
    print("   ✓ Image processor saved")

    print("\n2. Downloading model weights...")
    model = AutoModel.from_pretrained("raidium/curia")
    model.save_pretrained(pretrain_dir)
    print("   ✓ Model weights saved")

    # Verify files
    print("\n3. Verifying downloaded files...")
    expected_files = ["config.json", "preprocessor_config.json"]
    for file in expected_files:
        file_path = pretrain_dir / file
        if file_path.exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (missing)")

    # Check for model weights
    model_files = list(pretrain_dir.glob("*.bin")) + list(
        pretrain_dir.glob("*.safetensors")
    )
    if model_files:
        print(f"   ✓ Model weights: {[f.name for f in model_files]}")
    else:
        print(f"   ✗ No model weight files found")

    print(f"\n✓ Download complete!")
    print(f"✓ Model saved to: {pretrain_dir}")
    print(f"\nYou can now run feature caching with:")
    print(f"  poetry run diffbenchmark-cache model.name=curia dataset.name=camcan")


if __name__ == "__main__":
    try:
        download_curia_model()
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the model exists: https://huggingface.co/raidium/curia")
        print("3. If it's a private model, authenticate with: huggingface-cli login")
        raise
