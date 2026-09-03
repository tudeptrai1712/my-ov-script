"""
OpenVINO GenAI Model Manager Launcher.
Provides interactive and CLI management for OpenVINO models:
- Listing local models with disk usage and metadata
- Downloading OpenVINO IR models from Hugging Face
- Converting raw PyTorch / SafeTensors models to OpenVINO (optimum-intel)
- Deleting local models to free disk space

Usage:
    python manage_models.py               # Interactive menu
    python manage_models.py list          # List models
    python manage_models.py download <hf_repo_id>
    python manage_models.py convert <hf_repo_or_dir>
    python manage_models.py delete <model_name>
"""

from ovchat.verifier import verify_dependencies

if __name__ == "__main__":
    verify_dependencies(auto_install=True)
    from ovchat.manager import main
    main()

