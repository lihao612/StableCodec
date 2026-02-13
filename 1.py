from huggingface_hub import snapshot_download

sd_path = snapshot_download(
    repo_id="stabilityai/sd-turbo",
    local_dir="weights/sd-turbo",
    local_dir_use_symlinks=False
)
