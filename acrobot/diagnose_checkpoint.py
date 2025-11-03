import os
import torch

checkpoint_path = "./models/acrobot_checkpoints/15bf641c-dbc0-4f2f-b62f-fe04f568aacb/checkpoint_epoch1_step70000.pt"

print(f"Checking checkpoint: {checkpoint_path}")
print(f"File exists: {os.path.exists(checkpoint_path)}")

if os.path.exists(checkpoint_path):
    file_size = os.path.getsize(checkpoint_path)
    print(f"File size: {file_size} bytes ({file_size / (1024**2):.2f} MB)")
    
    # Check if file size is suspiciously small
    if file_size < 1000:
        print("WARNING: File size is very small, likely incomplete or corrupt")
        # Try to read the file as text to see what's in it
        try:
            with open(checkpoint_path, 'rb') as f:
                first_bytes = f.read(100)
                print(f"First 100 bytes: {first_bytes}")
        except Exception as e:
            print(f"Error reading file bytes: {e}")
    
    # Try loading with torch
    print("\nAttempting to load with torch.load...")
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        print("✓ Successfully loaded checkpoint!")
        print(f"Keys in checkpoint: {state.keys()}")
        
        if "model_state_dict" in state:
            print(f"✓ model_state_dict found")
            print(f"  Number of parameters: {len(state['model_state_dict'])}")
        else:
            print("✗ model_state_dict NOT found in checkpoint")
            
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to get more details about the file
        import zipfile
        try:
            with zipfile.ZipFile(checkpoint_path, 'r') as zf:
                print("\nZip file contents:")
                for name in zf.namelist():
                    info = zf.getinfo(name)
                    print(f"  {name}: {info.file_size} bytes")
        except zipfile.BadZipFile:
            print("\n✗ File is not a valid zip file (PyTorch checkpoints are zip files)")
        except Exception as zip_error:
            print(f"\n✗ Error reading as zip: {zip_error}")
            
        # Check if the file might be incomplete (still being written)
        print(f"\nFile permissions: {oct(os.stat(checkpoint_path).st_mode)}")
        print(f"Last modified: {os.path.getmtime(checkpoint_path)}")

else:
    print("✗ File does not exist!")

