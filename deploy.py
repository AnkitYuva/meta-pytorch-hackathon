from huggingface_hub import HfApi
import sys

try:
    api = HfApi(token="<YOUR_HUGGINGFACE_TOKEN_HERE>")
    api.upload_folder(
        folder_path=".",
        repo_id="yuvaankit/customer-support-openenv",
        repo_type="space",
        ignore_patterns=[
            ".git", "__pycache__", "*.pyc", 
            "test_smoke.py", "out.txt", "deploy.py", ".env"
        ]
    )
    print("SUCCESS: Code successfully deployed to Hugging Face Spaces!")
except Exception as e:
    print(f"FAILED TO DEPLOY: {e}")
    sys.exit(1)
