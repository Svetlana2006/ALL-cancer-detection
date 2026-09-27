import kagglehub
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("shafiullahshafin/c-nmc-2019-dataset")
print(f"Path to dataset files: {path}")

# List all files and directories in the downloaded path
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]: # just print first 5 files per directory to avoid huge output
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")
