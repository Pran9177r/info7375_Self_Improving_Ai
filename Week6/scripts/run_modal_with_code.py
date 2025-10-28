"""
Upload our tinyzero code and run training on Modal
"""
import modal
from pathlib import Path

# Create image
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    )
)

app = modal.App("tinyzero-training", image=image)
volume = modal.Volume.from_name("tinyzero-outputs", create_if_missing=True)

@app.function(
    gpu="A100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
)
def train(code_tar: bytes, config_yaml: str):
    """
    Run training with uploaded code
    """
    import subprocess
    import sys
    import os
    import tarfile
    import io
    
    print("="*60, flush=True)
    print(" TINYZERO TRAINING ON MODAL GPU", flush=True)
    print("="*60, flush=True)
    
    # Setup workspace
    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    # Extract code
    print("\n Extracting code...", flush=True)
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()
    print("✓ Code extracted", flush=True)
    
    # Write config
    print("\n Writing config...", flush=True)
    os.makedirs("configs", exist_ok=True)
    with open("configs/modal_config.yaml", "w") as f:
        f.write(config_yaml)
    print("✓ Config written", flush=True)
    
    # Install package
    print("\n Installing tinyzero...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                   check=True, capture_output=True)
    print("✓ Package installed", flush=True)
    
    # Run training with REAL-TIME output
    print("\n" + "="*60, flush=True)
    print(" STARTING TRAINING", flush=True)
    print("="*60, flush=True)
    sys.stdout.flush()
    
    # Run WITHOUT capturing output - streams directly to logs
    result = subprocess.run([
        sys.executable, 
        "-u",  # Unbuffered Python
        "-m", "tinyzero.train",
        "--config", "configs/modal_config.yaml",
        "--output_dir", "/outputs"
    ])
    
    if result.returncode != 0:
        raise Exception(f"Training failed with code {result.returncode}")
    
    print("\n" + "="*60, flush=True)
    print(" TRAINING COMPLETE!", flush=True)
    print("="*60, flush=True)
    
    volume.commit()
    
    return {"status": "success", "output_dir": "/outputs"}


@app.local_entrypoint()
def main():
    """Local entry point"""
    import tarfile
    import io
    from pathlib import Path
    
    print("\n" + "="*60)
    print(" PREPARING TINYZERO FOR MODAL")
    print("="*60)
    
    # Check we're in right directory
    if not Path("tinyzero").exists():
        print(" Error: Run from week_06/ directory!")
        return
    
    # Create tarball of code
    print("\n Packaging code...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add('tinyzero', arcname='tinyzero')
        tar.add('setup.py', arcname='setup.py')
    
    code_tar = tar_buffer.getvalue()
    print(f"✓ Packaged {len(code_tar)} bytes")
    
    # Load config
    print("\n Loading config...")
    with open("configs/modal_config.yaml") as f:
        config_yaml = f.read()
    print("✓ Config loaded")
    
    # Run on Modal
    print("\n" + "="*60)
    print(" LAUNCHING ON MODAL")
    print("="*60)
    
    result = train.remote(code_tar, config_yaml)
    
    print("\n" + "="*60)
    print(" DONE!")
    print("="*60)
    print(f"Result: {result}")
    print("\nTo download results:")
    print("modal volume get tinyzero-outputs /outputs ./results")
