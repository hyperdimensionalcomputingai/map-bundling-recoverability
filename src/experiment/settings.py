"""Edit these constants to create a different, explicitly named run."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "artifacts" / "demo"
DATA_DIR = ROOT / "data"
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL = "nomic-embed-text:latest"
MODEL_DIGEST = "0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f"
EMBEDDING_DIM = 768
DIMENSIONS = 4096
SEED = 2026
ROLES = ("age", "eye_color", "interest")
ENCODER_VERSION = "nomic-map-rademacher-v1"
PREFIXES = {"document": "search_document: ", "query": "search_query: "}
