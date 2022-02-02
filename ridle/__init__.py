import os
from pathlib import Path


def get_root_dir():
    """
    Get the root directory of the project.
    """
    return os.path.dirname(os.path.abspath(Path(__file__).parent))


ROOT_DIR = get_root_dir()
