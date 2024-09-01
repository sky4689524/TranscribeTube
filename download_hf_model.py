from huggingface_hub import snapshot_download

# Specify the repository ID and local directory where you want to download the model
repo_id = "desire hugging face model name" # For example , openai/whisper-large-v3
download_folder = ""  # Replace with your desired local folder path

# Download the entire repository into the specified local folder
snapshot_download(repo_id=repo_id, local_dir=download_folder)

print(f"Model repository downloaded to {download_folder}")