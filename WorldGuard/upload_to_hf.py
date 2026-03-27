"""Upload WorldGuard weights and config to Hugging Face Hub.

Usage:
    1. huggingface-cli login   (paste your HF token when prompted)
    2. python upload_to_hf.py
"""

from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "irfanalee/worldguard"

FILES = [
    "README.md",
    "configs/train_default.yaml",
    "configs/thresholds/ucsd.json",
    "checkpoints/train_default_epoch050_val0.0191.pt",
]


def main():
    api = HfApi()

    print(f"Creating repo: {HF_REPO_ID}")
    create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)

    for fpath in FILES:
        print(f"  Uploading {fpath} ...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fpath,
            repo_id=HF_REPO_ID,
        )

    print(f"\nDone! View at: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
