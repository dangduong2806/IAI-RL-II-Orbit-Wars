import os
import tarfile
import shutil

def create_submission():
    submission_dir = r"d:\Kaggle_final\submission"
    # Create submission directory
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
        
    # Copy hybrid_1k2.py as main.py (Entry point for Kaggle)
    source_script = r"d:\Kaggle_final\hybrid_1k2.py"
    target_script = os.path.join(submission_dir, "main.py")
    
    if os.path.exists(source_script):
        shutil.copy(source_script, target_script)
        print(f"Copied: {source_script} -> {target_script}")
    else:
        print(f"Error: {source_script} not found!")
        return

    # Choose the model weights to include
    source_model = r"d:\Kaggle_final\best_deep.pth"
    
    target_model = os.path.join(submission_dir, "best_deep.pth")
    
    if os.path.exists(source_model):
        shutil.copy(source_model, target_model)
        print(f"Copied: {source_model} -> {target_model}")
    else:
        print(f"Warning: No trained model (.pth) found yet. Please run the training script first!")
        # We still package it, but it will use random weights if .pth is missing.
        
    # Create the tar.gz file
    tar_path = r"d:\Kaggle_final\submission.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # Add main.py
        tar.add(target_script, arcname="main.py")
        
        # Add model if exists
        if os.path.exists(target_model):
            tar.add(target_model, arcname="best_deep.pth")
            
    print(f"\nSUCCESS! Submission packaged at: {tar_path}")
    print("Upload this file directly to Kaggle.")

if __name__ == "__main__":
    create_submission()
