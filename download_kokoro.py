import os
from huggingface_hub import snapshot_download, HfApi

print("Attempting to download Kokoro-82M model...")

# Try different possible repositories
repositories = [
    'kokoro-ai/Kokoro-82M',
    'agiresearch/Kokoro-82M', 
    'KwaiVGI/Kokoro-82M',
    'kokoro-ai/Kokoro-82M-v0'
]

api = HfApi()
success = False

foreach (repo in repositories):
    try:
        print(f"Checking repository: {repo}")
        
        # First check if repo exists
        model_info = api.model_info(repo)
        print(f"Repository exists: {repo}")
        print(f"Model ID: {model_info.modelId}")
        
        # Try to download
        print(f"Attempting download from: {repo}")
        snapshot_download(
            repo_id=repo,
            local_dir='./Kokoro-82M',
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.pt", "*.pth", "*.bin", "*.safetensors"]
        )
        print(f" Successfully downloaded from {repo}")
        success = True
        break
        
    except Exception as e:
        print(f" Failed with {repo}: {str(e)}")
        continue

if not success:
    print("")
    print("="*50)
    print("All repositories failed. Possible solutions:")
    print("1. The model might require authentication")
    print("2. Check if you need to accept terms of use")
    print("3. The model name might be different")
    print("")
    print("Searching for similar models...")
    try:
        models = api.list_models(search="kokoro")
        for model in list(models)[:5]:
            print(f"Found: {model.modelId}")
    except:
        print("Could not search for alternative models")
