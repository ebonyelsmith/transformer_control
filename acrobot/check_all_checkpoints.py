import os
import torch
import zipfile

checkpoint_dir = "./models/acrobot_checkpoints/15bf641c-dbc0-4f2f-b62f-fe04f568aacb"

print(f"Checking all checkpoints in: {checkpoint_dir}\n")

checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])

print(f"Found {len(checkpoint_files)} checkpoint files\n")

results = []

for ckpt_file in checkpoint_files:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
    file_size = os.path.getsize(ckpt_path)
    
    # Try to load
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        status = "✓ OK"
        has_model_state = "model_state_dict" in state
        results.append((ckpt_file, file_size, status, has_model_state))
    except Exception as e:
        status = f"✗ FAILED: {str(e)[:60]}"
        results.append((ckpt_file, file_size, status, False))

# Print results
print(f"{'Checkpoint':<45} {'Size (MB)':<12} {'Status':<20}")
print("=" * 100)
for ckpt_file, size, status, has_model in results:
    size_mb = size / (1024**2)
    print(f"{ckpt_file:<45} {size_mb:>10.2f}  {status}")

# Summary
ok_count = sum(1 for _, _, status, _ in results if status == "✓ OK")
failed_count = len(results) - ok_count

print(f"\n{'='*100}")
print(f"Summary: {ok_count} OK, {failed_count} FAILED out of {len(results)} total checkpoints")

if failed_count > 0:
    print("\nFailed checkpoints:")
    for ckpt_file, size, status, _ in results:
        if status != "✓ OK":
            print(f"  - {ckpt_file}")

