import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PARENT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = os.path.join(PARENT_DIR, "data")

# Get the parent directory of the data directory
CACHE_DIR = os.getenv("CACHE_DIR")

